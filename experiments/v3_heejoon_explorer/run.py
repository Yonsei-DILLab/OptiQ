"""One independent 1M run; online W&B and full-state latest checkpoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import time
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import jax
import wandb
from flax import serialization
import run_optiq_dime as production
from .algorithm import ExplorerOptiQ, EvaluatorCallback, Paused
from . import checkpoint

ROOT=Path(__file__).resolve().parents[2]


def write(path, data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def config(env, seed, out, smoke=False):
    with hydra.initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        overrides=[f'benchmark={env}', f'seed={seed}', f'output_root={out}',
                   f'run_name={env}_v3_heejoon_explorer_s{seed}']
        if smoke: overrides+=['total_steps=128','alg.learning_starts=32','alg.actor.learning_starts=32',
                              'alg.buffer_size=256','eval_interval=64','num_eval_episodes=1']
        cfg=hydra.compose(config_name='v3_heejoon_explorer',overrides=overrides)
    production.validate_config(cfg)
    assert cfg.alg.actor.num_policy_samples==16 and cfg.alg.actor.proposals_per_policy_sample==4
    assert cfg.alg.actor.density_beta==1 and not cfg.alg.actor.include_anchor
    return cfg


def make_model(cfg):
    original, cb = production.OptiQDIME, production.MujocoEvalCallback
    try:
        production.OptiQDIME=ExplorerOptiQ;production.MujocoEvalCallback=EvaluatorCallback
        model,callbacks=production.create_algorithm(cfg)
    finally:
        production.OptiQDIME=original;production.MujocoEvalCallback=cb
    model.model_save_path=None  # Full resume checkpoints below, not legacy weights-only saves.
    return model,callbacks


def verify_source():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    bad=[p for p,h in manifest['files'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not bad,bad
    return manifest


def verify_online(run, step):
    token=int(time.time()*1000);run.summary['online_probe']=token
    api=wandb.Api(timeout=15);deadline=time.time()+50
    while time.time()<deadline:
        try:
            api.flush();remote=api.run(f'{run.entity}/{run.project}/{run.id}')
            if remote.summary.get('online_probe')==token:return
        except Exception:pass
        time.sleep(2)
    raise RuntimeError(f'W&B upload was not confirmed at step {step}')


def run(args):
    manifest=verify_source();assert jax.default_backend()=='gpu' and len(jax.devices())==1
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    cfg=config(args.env,args.seed,str(out),args.smoke)
    commit=manifest['commit'];rid=hashlib.sha256((commit+args.env+str(args.seed)+str(args.smoke)).encode()).hexdigest()[:12]
    resolved=OmegaConf.to_container(cfg,resolve=True)
    resolved['runtime']=dict(commit=commit,source_code_id=manifest['source_code_id'],hostname=socket.gethostname(),
                             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                             packages={k:importlib.metadata.version(k) for k in ['jax','flax','mujoco','gymnasium','wandb']})
    resolved['roles']=dict(collection='explorer',ot_source='explorer',kde_centers='explorer',
                           evaluation='selective_evaluator',td_action='selective_evaluator',td_q='target_twin_min',
                           extraction_q='live_twin_mean',evaluator_update='positive_advantage_weighted_action_mse',
                           advantage_mode='target_min_minus_target_max_same_z',evaluator_weight='positive_A',
                           evaluator_loss_normalization='accepted_count',
                           extra_td_noise=False,trust_region=False)
    write(out/'config.json',resolved)
    run=wandb.init(entity=cfg.wandb.entity,project=cfg.wandb.project,group=cfg.wandb.group,
                   id=rid,resume='allow',name=cfg.run_name,config=resolved,mode='online',dir=str(out),save_code=False,
                   job_type='validation' if args.smoke else 'train',tags=['legacy','explorer-evaluator',args.env],
                   settings=wandb.Settings(init_timeout=90))
    run.define_metric('env_steps');run.define_metric('*',step_metric='env_steps')
    model=callbacks=None;failed=True;start=time.time()
    try:
        verify_online(run,0)
        write(out/'wandb.json',dict(id=run.id,url=run.url,project=run.project,entity=run.entity,verified=True))
        model,callbacks=make_model(cfg);path=out/'resume.zip';resumed=path.exists()
        if resumed:checkpoint.load(model,callbacks,path,commit)
        last_save=model.num_timesteps

        def hook(m):
            nonlocal last_save
            if m.num_timesteps%1000==0:
                write(out/'progress.json',dict(step=m.num_timesteps,updates=m._n_updates,
                                               elapsed_seconds=time.time()-start,time=time.time(),wandb_url=run.url,
                                               evaluator_gradient_steps=int(m.policy.target_actor_state.step)))
            if m.num_timesteps-last_save>=50000 or m.stop_requested:
                receipt=checkpoint.save(m,callbacks,path,commit);last_save=m.num_timesteps
                write(out/'checkpoint.json',receipt)
        model.checkpoint_hook=hook
        def stop(*_):model.stop_requested=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        write(out/'RUNNING.json',dict(pid=os.getpid(),env=args.env,seed=args.seed,commit=commit,
                                      started=time.time(),resumed=resumed,start_step=model.num_timesteps,wandb_url=run.url))
        try:
            model.learn(total_timesteps=int(cfg.total_steps)-model.num_timesteps,reset_num_timesteps=not resumed,
                        progress_bar=False,callback=callbacks,log_interval=1,tb_log_name='v3_heejoon_explorer')
        except Paused:pass
        if model.logger.name_to_value:model.logger.dump(model.num_timesteps)
        receipt=checkpoint.save(model,callbacks,path,commit);write(out/'checkpoint.json',receipt)
        done=model.num_timesteps>=cfg.total_steps
        write(out/'progress.json',dict(step=model.num_timesteps,updates=model._n_updates,complete=done,
                                       elapsed_seconds=time.time()-start,wandb_url=run.url))
        run.summary.update(dict(completed=done,timesteps=model.num_timesteps,updates=model._n_updates,
                                evaluator_gradient_steps=int(model.policy.target_actor_state.step)))
        verify_online(run,model.num_timesteps)
        if done:
            (out/'final_models.msgpack').write_bytes(serialization.to_bytes(checkpoint.states(model)))
            write(out/'COMPLETE.json',dict(**receipt,commit=commit,wandb_url=run.url,finished=time.time()))
        failed=False
    except BaseException as exc:
        if model is not None and getattr(model,'_last_obs',None) is not None:
            try:checkpoint.save(model,callbacks,out/'resume.zip',commit)
            except Exception as ce:print('CHECKPOINT_FAILED',repr(ce),flush=True)
        write(out/'FAILED.json',dict(error=repr(exc),time=time.time(),commit=commit));raise
    finally:
        if model is not None:model.get_env().close();model.logger.close()
        if callbacks:
            for cb in callbacks.callbacks:
                if hasattr(cb,'eval_env'):cb.eval_env.close()
        run.finish(exit_code=int(failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['ant','humanoid','halfcheetah'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
