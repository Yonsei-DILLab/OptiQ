"""Frozen center-start mu-only evaluation and conditional short-horizon Q probe."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--turning-reference',type=Path)
    ap.add_argument('--spatial-q',action='store_true')
    a=ap.parse_args()
    report_source=Path(__file__).resolve().parents[1]
    report_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=report_source,text=True).strip()
    cfg=json.loads((a.run/'config.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()==cfg['source_commit']
    checkpoint=a.run/'checkpoint-final.pt'
    digest=sha(checkpoint)
    assert digest==json.loads((a.run/'checkpoint-verification.json').read_text())['sha256']
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    import flax.serialization as fs
    import gymnasium as gym
    from omegaconf import OmegaConf
    from antmaze_experiments.learners import JaxLearner, evaluation_rng
    from antmaze_experiments.envs import make_one
    from antmaze_experiments.run import evaluate
    torch.set_num_threads(1)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['step']==100000 and payload['updates']==90000
    class Descriptor(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(29,),dtype=np.float32)
        action_space=gym.spaces.Box(-1,1,shape=(8,),dtype=np.float32)
    spec=importlib.util.spec_from_file_location('frozen_trg',a.source/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=OmegaConf.create(cfg['native']);native.output_root=str(a.output/'constructor')
    model=module.runner.OptiQDIME('MlpPolicy',env=Descriptor(),cfg=native,model_save_path=None,save_every_n_steps=100000)
    p=model.policy
    state=fs.from_bytes(dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state),payload['learner']['policy'])
    p.actor_state=state['actor'];p.qf_state=state['critic'];p.target_actor_state=state['target_actor']
    before=fs.to_bytes(state)
    p.key=jax.random.PRNGKey(91000);p.noise_key=jax.random.PRNGKey(91001)
    learner=JaxLearner.__new__(JaxLearner)
    learner.method='optiq';learner.model=model;learner.reward_profile=cfg['reward_profile']
    learner.eval_random_starts=False;learner.eval_fixed_starts=True
    if a.spatial_q:
        from .spatial_q_probe import diagnose
        result=diagnose(learner,a.output)
        assert fs.to_bytes(dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state))==before
        assert sha(checkpoint)==digest
        result.update(checkpoint_and_model_unchanged=True,training_source=cfg['source_commit'],
                      report_source=report_commit,checkpoint_sha256=digest)
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    if a.turning_reference is not None:
        from .turning_v1_probe import diagnose
        result=diagnose(learner,a.turning_reference,a.output)
        assert fs.to_bytes(dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state))==before
        assert sha(checkpoint)==digest
        result.update(checkpoint_and_model_unchanged=True,training_source=cfg['source_commit'],
                      report_source=report_commit,checkpoint_sha256=digest)
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    summary=evaluate(learner,'v1',a.output,100000,100,'native',fixed=True)
    data=np.load(a.output/'evaluations/0000100000/native-fixed/rollouts.npz')
    np.testing.assert_array_equal(data['initial_full_state'],np.repeat(data['initial_full_state'][:1],100,axis=0))
    np.testing.assert_array_equal(data['initial_full_state'][:,:2],np.zeros((100,2)))
    paths=[v[np.isfinite(v).all(axis=1)] for v in data['xy']]
    routes=[]
    for v in paths:
        hit=np.flatnonzero(v[:,0]<=-4)
        routes.append(('upper' if v[hit[0],1]>0 else 'lower') if len(hit) else 'unresolved')
    counts={k:routes.count(k) for k in ('upper','lower','unresolved')}
    result=dict(rollout_summary=summary,routes=counts,
                training_source=cfg['source_commit'],report_source=report_commit,
                checkpoint_sha256=digest,conditional_noise=False,random_z_each_action=True,
                exact_identical_center_full_state=True)
    print(json.dumps(result),flush=True)
    if counts['lower']>counts['upper']:
        env=make_one('v1',0,reward_profile=cfg['reward_profile'],random_init=False)
        obs=env.reset();initial=env.state()
        with evaluation_rng(learner,91002):
            actor_actions=learner.act(np.repeat(obs[None],512,axis=0),'native')
        uniform=np.random.default_rng(91003).uniform(-1,1,(512,8)).astype(np.float32)
        candidates=np.concatenate((actor_actions,uniform))
        observations=jnp.asarray(np.repeat(obs[None],len(candidates),axis=0))
        qs=np.asarray(p.qf_state.apply_fn({'params':p.qf_state.params,'batch_stats':p.qf_state.batch_stats},observations,jnp.asarray(candidates),train=False))[...,0].T
        assert qs.shape==(1024,2)
        displacements=[];short_returns=[]
        # Classify physical direction, never assume a joint torque sign means up/down.
        # Constant-action10-step displacement is a LOCAL probe, not a route-value oracle.
        for action in candidates:
            env.restore(initial);total=0.
            for t in range(10):
                nxt,r,done,info=env.step(action);total+=float(model.gamma)**t*float(r)
                if done:break
            displacements.append(nxt[:2]-obs[:2]);short_returns.append(total)
        delta=np.asarray(displacements)
        np.savez_compressed(a.output/'center_action_probe.npz',actions=candidates,q=qs,delta_xy=delta,short_returns=short_returns,actor_count=512)
        stats={}
        for name,sl in [('actor',slice(0,512)),('uniform',slice(512,1024))]:
            dy=delta[sl,1];q=qs[sl].mean(axis=1)
            stats[name]={}
            for direction,mask in [('up',dy>1e-3),('down',dy<-1e-3),('near_zero',np.abs(dy)<=1e-3)]:
                stats[name][direction]=dict(count=int(mask.sum()),q_mean=float(q[mask].mean()) if mask.any() else None)
            best=np.argsort(q)[-51:]
            stats[name]['top10pct_q_up_fraction']=float(np.mean(dy[best]>1e-3))
        result['local_probe']=stats
        result['local_probe_caveat']='Direction is 10-step constant-action delta y, not long-horizon route; uniform candidates may be OOD.'
        env.close()
    assert fs.to_bytes(dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state))==before
    assert sha(checkpoint)==digest
    result['checkpoint_and_model_unchanged']=True
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
