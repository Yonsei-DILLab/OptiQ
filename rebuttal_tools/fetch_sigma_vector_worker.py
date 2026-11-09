"""Isolated Fetch scale ablation using a pinned, unchanged iBOLT learner."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

VARIANTS = dict(narrow=(-5., -3.5, -4.), wide=(-5., 1., -1.), original=(-5., -1., -1.))


def atomic(path, data):
    path = Path(path);temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n');temp.replace(path)


def run(spec, smoke=False, device='cuda'):
    source = Path(spec['source']).resolve();sys.path.insert(0, str(source))
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=source, text=True).strip()
    assert branch and 'anonymous' not in branch.lower()
    manifest = json.loads(Path(spec['source_manifest']).read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize
    import train
    from exploration import ExplorationIBOLT
    from fetch_rebuttal.run import IBOLTView, guard, record_evaluation
    from fetch_rebuttal.env import make_env
    from fetch_rebuttal.normalization import normalized_ibolt_class
    from rebuttal_tools.fetch_vector_env import GroupedFetchVecEnv
    from rebuttal_tools.fetch_vector_replay import VectorHERReplayBuffer
    guard()
    if device=='cuda':assert jax.default_backend()=='gpu'
    seed=spec['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    n_envs=256;warmup=12800;total=13312 if smoke else 1000192;gradients=32
    expected=(total-warmup)//8
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoints','evaluations','learner'):(output/name).mkdir()
    started=time.time();model=vec=reward_env=None
    settings=dict(method='ibolt',task='pick-and-place',seed=seed,temperature=.25,
        steps=total,target_steps=1000000,warmup=warmup,batch_size=256,num_envs=n_envs,utd=.125,
        gradient_steps_per_vector_step=gradients,replay_capacity=25600 if smoke else 1000000,
        eval_every=5000,eval_episodes=2 if smoke else 10,final_episodes=2 if smoke else 100,
        checkpoint_every=100000,HER=True,normalize_observations=True,variant=spec['variant'],
        log_std_min=VARIANTS[spec['variant']][0],log_std_max=VARIANTS[spec['variant']][1],
        initial_log_std=VARIANTS[spec['variant']][2],smoke=smoke,
        primary_success='Final-step success with fresh stochastic z and mu only',
        expected_updates=expected,worker_cpus=spec['worker_cpus'],source_manifest=spec['source_manifest'])
    atomic(output/'config.json',settings)
    atomic(output/'progress.json',dict(status='initializing',steps=0,updates=0,observed_unix=time.time()))
    try:
        cfg=train.compose_config(['benchmark=ant','wandb.activate=false'])
        cfg.env_name='FetchPickAndPlace-v4';cfg.task='pick-and-place';cfg.seed=seed;cfg.temperature=.25
        cfg.total_steps=total;cfg.output_root=str(output);cfg.require_gpu=device=='cuda'
        cfg.alg.learning_starts=cfg.alg.actor.learning_starts=warmup
        cfg.alg.batch_size=256;cfg.alg.buffer_size=settings['replay_capacity'];cfg.alg.utd=.125
        # Run every original validation on the original bounds before installing
        # the explicitly authorized experimental scale bounds. Learner bytes stay pinned.
        train.validate_config(cfg)
        low,high,initial=VARIANTS[spec['variant']]
        assert all(math.isfinite(x) for x in (low,high,initial)) and low<=initial<=high
        cfg.alg.actor.log_std_min=low;cfg.alg.actor.log_std_max=high;cfg.alg.actor.initial_log_std=initial
        assert list(cfg.alg.actor.hidden_dims)==[256,256] and list(cfg.alg.critic.hs)==[256,256]
        assert cfg.alg.actor.num_policy_samples==64 and cfg.alg.actor.proposals_per_policy_sample==1
        atomic(output/'mujoco-config.json',OmegaConf.to_container(cfg,resolve=True))
        vec=VecNormalize(GroupedFetchVecEnv(n_envs,spec['worker_cpus']),norm_obs=True,norm_reward=False,clip_obs=5.,epsilon=1e-8)
        reward_env=make_env('pick-and-place')
        cls=normalized_ibolt_class(ExplorationIBOLT)
        model=cls('MlpPolicy',env=vec,cfg=cfg,model_save_path=str(output/'checkpoints'),
            save_every_n_steps=100000,gradient_steps=gradients,replay_buffer_class=VectorHERReplayBuffer,
            replay_buffer_kwargs=dict(compute_reward=reward_env.unwrapped.compute_reward,her_seed=seed,
                table_height=float(reward_env.unwrapped.height_offset)))
        model.set_logger(configure(str(output/'learner'),['csv']))
        view=IBOLTView(model)
        # Height strata use the same fixed environment resets as native evaluation.
        python_state,numpy_state=random.getstate(),np.random.get_state()
        initial=np.stack([reward_env.reset(seed=700000+seed*1000+i)[0] for i in range(settings['final_episodes'])])
        random.setstate(python_state);np.random.set_state(numpy_state)
        heights=initial[:,-1]-initial[:,5]
        masks=dict(ground=np.abs(heights)<.001,low_air=(heights>=.001)&(heights<=.05),air_over5cm=heights>.05)
        def evaluate(step):
            record=record_evaluation(view,settings,output,step,model._n_updates,ibolt=True)
            strata=dict(steps=step,modes={})
            for mode in ('stochastic_z','zero_z'):
                values=record[mode];count=values['episodes'];hits=np.asarray(values['final_success']);anyhits=np.asarray(values['any_success'])
                strata['modes'][mode]={name:dict(episodes=int(mask[:count].sum()),
                    final_successes=int(hits[mask[:count]].sum()),any_successes=int(anyhits[mask[:count]].sum())) for name,mask in masks.items()}
            with (output/'height-evaluations.jsonl').open('a') as stream:stream.write(json.dumps(strata)+'\n')
            return record
        next_log=warmup;next_checkpoint=100000
        original_train=model.train
        def measured_train(batch_size,gradient_steps):
            nonlocal next_log,next_checkpoint
            assert batch_size==256 and gradient_steps==32
            original_train(batch_size,gradient_steps)
            step=int(model.num_timesteps)
            assert model._n_updates==(step-warmup)//8
            assert int(model.policy.actor_state.step)==model._n_updates
            metrics={}
            for k,v in model.logger.name_to_value.items():
                if isinstance(v,(int,float,np.number)):
                    value=float(v)
                    if k.startswith('train/') and not math.isfinite(value):raise FloatingPointError(k)
                    if math.isfinite(value):metrics[k]=value
            if step>=next_log or step==total:
                atomic(output/'progress.json',dict(status='running',steps=step,updates=model._n_updates,
                    actor_updates=int(model.policy.actor_state.step),elapsed_seconds=time.time()-started,
                    observed_unix=time.time(),backend=jax.default_backend(),utd=.125,metrics=metrics,
                    her_replay_last_sample=model.replay_buffer.last))
                model.logger.dump(step);next_log=step+4096
            if step>=next_checkpoint:
                model._save_model()
                while next_checkpoint<=step:next_checkpoint+=100000
        model.train=measured_train
        class Evaluation(BaseCallback):
            def __init__(self):super().__init__();self.next_eval=5000
            def _on_step(self):
                step=int(self.num_timesteps)
                if step%4096==0 and step<=warmup:
                    atomic(output/'progress.json',dict(status='warmup',steps=step,updates=0,
                        elapsed_seconds=time.time()-started,observed_unix=time.time()))
                if not smoke and step>=self.next_eval and step<total:
                    evaluate(step)
                    while self.next_eval<=step:self.next_eval+=5000
                return True
        evaluate(0)
        model.learn(total_timesteps=total,callback=Evaluation(),log_interval=100000000)
        assert model.num_timesteps==total and model._n_updates==expected
        assert int(model.policy.actor_state.step)==expected
        for state in (model.policy.actor_state,model.policy.qf_state):
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        model._save_model()
        if not smoke:model.save_replay_buffer(output/'checkpoints/final-replay.pkl')
        final=evaluate(total)
        obs=jnp.zeros((16,28));z=jax.random.normal(jax.random.PRNGKey(781),(16,4))
        _,ls=model.policy.actor_state.apply_fn({'params':model.policy.actor_state.params},obs,z)
        assert np.asarray(ls).min()>=low-1e-5 and np.asarray(ls).max()<=high+1e-5
        result=dict(completed=True,steps=total,updates=expected,actor_updates=expected,
            observed_utd=(expected/(total-warmup)),backend=jax.default_backend(),
            elapsed_seconds=time.time()-started,finished_unix=time.time(),variant=spec['variant'],seed=seed,
            final_success=final['stochastic_z']['success_rate'],smoke=smoke,
            logstd_probe=dict(min=float(ls.min()),max=float(ls.max())),her=model.replay_buffer.last)
        atomic(output/'completed.json',result);atomic(output/'progress.json',dict(result,status='completed'))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),observed_unix=time.time()))
        raise
    finally:
        if vec is not None:vec.close()
        if reward_env is not None:reward_env.close()
        if model is not None:model.logger.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
