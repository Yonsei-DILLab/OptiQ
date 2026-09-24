"""Inference-only original-origin evaluation of immutable OptiQ checkpoints."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--episodes',type=int,default=40)
    ap.add_argument('--cpu-offset',type=int,default=0)
    ap.add_argument('--v1-origin-supplement',action='store_true',
                    help='Explicit supplementary v1 origin probe; primary native random evaluation remains unchanged')
    args=ap.parse_args()
    report_source=Path(__file__).resolve().parents[1]
    report_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=report_source,text=True).strip()
    cfg=json.loads((args.run/'config.json').read_text())
    if args.v1_origin_supplement:
        assert cfg['task']=='v1'
    else:
        assert cfg['task'] in ('v2','v3','v4'), 'v1 origin probes require the explicit supplementary flag'
    source=Path('/home/heechan/OptiQ-ops/sources')/cfg['source_commit']
    assert cfg['method']=='optiq' and source.exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==cfg['source_commit']
    assert args.checkpoint.is_relative_to(args.run)
    proof_path=(args.checkpoint.parent/'verification.json' if args.checkpoint.name=='policy.pt'
                else args.run/'checkpoint-verification.json')
    proof=json.loads(proof_path.read_text());assert proof['readback_verified']
    expected_hash=proof['sha256'];assert sha(args.checkpoint)==expected_hash
    assert args.episodes>0
    assert not os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('JAX_PLATFORMS')=='cpu'
    cpus=sorted(os.sched_getaffinity(0));offset=args.cpu_offset%len(cpus)
    os.sched_setaffinity(0,[cpus[(offset+i)%len(cpus)] for i in range(min(4,len(cpus)))])
    args.output.mkdir(parents=True,exist_ok=False)
    # Policy/environment imports come from the immutable training source.
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    import jax
    import flax.serialization as fs
    import gymnasium as gym
    from omegaconf import OmegaConf
    from antmaze_experiments.learners import JaxLearner
    from antmaze_experiments.progress_reward import PROFILES, distance, progress_scale, step_cost, bonus_enabled
    torch.set_num_threads(1)
    assert all(device.platform=='cpu' for device in jax.devices())
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert payload['config']==cfg
    assert payload['step']==(proof['step'] if 'step' in proof else proof['steps'])
    step=int(payload['step']);updates=int(payload['updates'])
    class Descriptor(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(29,),dtype=np.float32)
        action_space=gym.spaces.Box(-1,1,shape=(8,),dtype=np.float32)
        def reset(self,**kwargs):raise RuntimeError('Inference descriptor only')
        def step(self,action):raise RuntimeError('Inference descriptor only')
    spec=importlib.util.spec_from_file_location('frozen_trg_train',source/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=OmegaConf.create(cfg['native']);native.output_root=str(args.output/'constructor')
    model=module.runner.OptiQDIME('MlpPolicy',env=Descriptor(),cfg=native,model_save_path=None,save_every_n_steps=cfg['steps'])
    policy=model.policy
    template=dict(actor=policy.actor_state,critic=policy.qf_state,target_actor=policy.target_actor_state)
    saved=fs.from_bytes(template,payload['learner']['policy'])
    original=fs.msgpack_restore(payload['learner']['policy'])
    def equal(actual,wanted):
        if isinstance(wanted,dict):
            assert set(actual)==set(wanted)
            for key in wanted:equal(actual[key],wanted[key])
        elif isinstance(wanted,(list,tuple)):
            assert len(actual)==len(wanted)
            for x,y in zip(actual,wanted):equal(x,y)
        else:np.testing.assert_array_equal(np.asarray(actual),np.asarray(wanted))
    equal(fs.to_state_dict(saved),original)
    policy.actor_state=saved['actor'];policy.qf_state=saved['critic'];policy.target_actor_state=saved['target_actor']
    # Evaluation deliberately has its own seed, just like the frozen evaluator.
    # Saved optional/typed RNG placeholders may serialize as NumPy object arrays;
    # they are not parameters and must not be coerced into numeric JAX arrays.
    policy.key=jax.random.PRNGKey(700000+step)
    policy.noise_key=jax.random.PRNGKey(700000+step)
    def states():return dict(actor=policy.actor_state,critic=policy.qf_state,target_actor=policy.target_actor_state)
    before=hashlib.sha256(fs.to_bytes(states())).hexdigest()
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(states()))
    learner=JaxLearner.__new__(JaxLearner)
    learner.method='optiq';learner.model=model;learner.reward_profile=cfg['reward_profile']
    learner.eval_random_starts=False;learner.eval_fixed_starts=True
    # Only the evaluation loop comes from this new reporting source. Its relative
    # env/learner imports still resolve to the original training implementation.
    spec=importlib.util.spec_from_file_location('antmaze_experiments.fixed_evaluation_impl',report_source/'antmaze_experiments/run.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    write(args.output/'provenance.json',dict(training_source=cfg['source_commit'],evaluation_source=report_sha,
        parent_run=str(args.run),checkpoint=str(args.checkpoint),checkpoint_sha256=expected_hash,
        checkpoint_steps=step,updates=updates,episodes_per_mode=args.episodes,task=cfg['task'],training_seed=cfg['seed'],
        training_config=cfg,reset='original origin [0,0], original pose/velocity, identical full state across all episodes',
        evaluation_matches_training_resets=not args.v1_origin_supplement,
        supplementary_origin_probe=args.v1_origin_supplement,
        replaces_primary_evaluation=False,eval_starts='fixed',
        primary_trajectory=None if args.v1_origin_supplement else 'policy-fixed',
        modes=['policy','native'],policy='fresh random latent and conditional sigma; original sampler',
        native='fresh random latent, mu-only',evaluation_rng_seed=700000+step,
        external_noise=False,intrinsic_reward=False,inference_only=True))
    results={};full_starts=[];errors=[];started=time.monotonic()
    for mode in ('policy','native'):
        results[mode]=evaluator.evaluate(learner,cfg['task'],args.output,step,args.episodes,mode,fixed=True)
        file=args.output/'evaluations'/f'{step:010d}'/(mode+'-fixed')/'rollouts.npz'
        with np.load(file) as data:
            initial=data['initial_full_state'];full_starts.append(initial.copy())
            np.testing.assert_array_equal(initial,np.repeat(initial[:1],args.episodes,axis=0))
            np.testing.assert_array_equal(initial[:,:2],np.zeros((args.episodes,2)))
            for xy,n,g,ret in zip(data['xy'],data['lengths'],data['goals'],data['returns']):
                path=xy[:int(n)+1];assert np.isfinite(path).all()
                profile=cfg['reward_profile'];task=cfg['task']
                bonus=(20 if task=='v2' and g==1 else 10) if g else 0
                if profile in PROFILES:
                    expected=progress_scale(profile)*(distance(path[0],task,profile)-distance(path[-1],task,profile))-step_cost(profile)*n+(bonus if bonus_enabled(profile) else 0)
                    errors.append(abs(float(ret-expected)))
            if errors:assert max(errors)<.01,max(errors)
        write(args.output/'progress.json',dict(mode=mode,completed=True,step=step,seconds=time.monotonic()-started))
    np.testing.assert_array_equal(*full_starts)
    assert hashlib.sha256(fs.to_bytes(states())).hexdigest()==before
    assert sha(args.checkpoint)==expected_hash
    verification=dict(passed=True,training_checkpoint_unchanged=True,model_optimizer_unchanged=True,
        restored_policy_exact=True,identical_initial_full_state=True,initial_xy=[0.,0.],paired_modes=True,
        max_reward_sum_error=max(errors) if errors else None,
        raw_sha256={str(p.relative_to(args.output)):sha(p) for p in args.output.glob('evaluations/*/*/rollouts.npz')})
    write(args.output/'verification.json',verification)
    # A separate evaluation run prevents mixing corrected reset distributions
    # into the live training process's historical random-start metric series.
    try:
        import wandb
        parent_meta=json.loads((args.run/'wandb.json').read_text())
        prefix='v1-origin-supplement:' if args.v1_origin_supplement else 'fixed-origin:'
        identifier=hashlib.sha256((prefix+str(args.run)).encode()).hexdigest()[:12]
        group='antmaze-v1-origin-supplement-20260925' if args.v1_origin_supplement else 'antmaze-matched-start-eval-20260924'
        wb=wandb.init(entity='OptiQ',project='antmaze',group=group,
            id=identifier,resume='allow',name=args.run.name+'-origin-supplement' if args.v1_origin_supplement else args.run.name+'-fixed-eval',job_type='evaluation',mode='online',
            config=dict(training_source=cfg['source_commit'],evaluation_source=report_sha,
                parent_run_id=parent_meta['id'],parent_run_url=parent_meta['url'],task=cfg['task'],
                reward_profile=cfg['reward_profile'],eval_starts='original fixed origin/full state',training_seed=cfg['seed'],
                supplementary_origin_probe=args.v1_origin_supplement,replaces_primary_evaluation=False),
            settings=wandb.Settings(init_timeout=30))
        wb.define_metric('checkpoint_step')
        wb.define_metric('fixed/*',step_metric='checkpoint_step')
        metrics={'checkpoint_step':step,'episodes_per_mode':args.episodes}
        for mode,result in results.items():
            metrics.update({f'fixed/{mode}/success_rate':result['success_rate'],f'fixed/{mode}/return':result['mean_return']})
            for goal,count in result['goal_counts'].items():metrics[f'fixed/{mode}/goal_{goal}']=count
        wb.log(metrics)
        write(args.output/'wandb.json',dict(id=identifier,url=wb.url,project='OptiQ/antmaze',parent_run_id=parent_meta['id']))
        wb.finish()
    except Exception as error:
        write(args.output/'wandb-failure.json',dict(type=type(error).__name__,error=str(error),rollouts_verified=True))
    write(args.output/'result.json',dict(completed=True,step=step,evaluation_source=report_sha,training_source=cfg['source_commit'],
        checkpoint_sha256=expected_hash,results=results,seconds=time.monotonic()-started))
    print(json.dumps(dict(run=args.run.name,step=step,results=results)),flush=True)


if __name__=='__main__':main()
