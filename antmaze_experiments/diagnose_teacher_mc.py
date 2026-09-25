"""Frozen-checkpoint Monte Carlo diagnostics; never update a model or environment."""
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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    source = Path(__file__).resolve().parents[1]
    reporting_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    cfg = json.loads((args.run / 'config.json').read_text())
    result = json.loads((args.run / 'result.json').read_text())
    assert result['completed'] and cfg['task'] in ('v3', 'v4') and cfg['method'] == 'optiq'
    proof = json.loads((args.run / 'checkpoint-verification.json').read_text())
    checkpoint = args.run / 'checkpoint-final.pt'
    assert proof['readback_verified'] and proof['steps'] == 258304
    assert sha(checkpoint) == proof['sha256']
    frozen = Path('/home/heechan/OptiQ-ops/sources') / cfg['source_commit']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=frozen, text=True).strip() == cfg['source_commit']
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [cpus[(48+i) % len(cpus)] for i in range(min(4, len(cpus)))])
    sys.path.insert(0, str(frozen))
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    import flax.serialization as fs
    import gymnasium as gym
    from omegaconf import OmegaConf
    from antmaze_experiments.critic_diagnostics import evaluate_q, teacher_probe
    torch.set_num_threads(1)
    assert all(x.platform == 'cpu' for x in jax.devices())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['config'] == cfg and payload['step'] == proof['steps']
    replay = payload['replay']['buf_obs'].numpy()
    assert replay.shape == (258304, 29) and np.isfinite(replay).all()
    targets = ([(0,0),(-1,1),(1,-1),(-4,2),(4,-2),(-6,6),(6,-6)] if cfg['task']=='v3'
               else [(0,0),(-2,1.5),(-2,-1.5),(-5,4),(-5,-4),(-8,4),(-8,-4)])
    indices = [int(np.square(replay[:,:2]-xy).sum(-1).argmin()) for xy in targets]
    indices.append(int(np.random.default_rng(251091).integers(8192, len(replay))))
    observations = replay[indices].copy()
    selection = [dict(index=i, requested_xy=list(xy), actual_xy=replay[i,:2].tolist(),
                      distance=float(np.linalg.norm(replay[i,:2]-xy))) for i,xy in zip(indices,targets)]
    selection.append(dict(index=indices[-1],requested_xy=None,actual_xy=observations[-1,:2].tolist()))

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(29,),dtype=np.float32)
        action_space = gym.spaces.Box(-1,1,shape=(8,),dtype=np.float32)
        def reset(self, **kwargs):
            raise RuntimeError('Descriptor only')
        def step(self, action):
            raise RuntimeError('Descriptor only')

    spec = importlib.util.spec_from_file_location('frozen_trg', frozen/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native = OmegaConf.create(cfg['native']);native.output_root = str(args.output/'constructor')
    model = module.runner.OptiQDIME('MlpPolicy',env=Descriptor(),cfg=native,
                                  model_save_path=None,save_every_n_steps=cfg['steps'])
    policy = model.policy
    template = dict(actor=policy.actor_state,critic=policy.qf_state,target_actor=policy.target_actor_state)
    states = fs.from_bytes(template,payload['learner']['policy'])
    original = fs.msgpack_restore(payload['learner']['policy'])
    def equal(a,b):
        if isinstance(b,dict):
            assert set(a)==set(b)
            for k in b:equal(a[k],b[k])
        else:np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    equal(fs.to_state_dict(states),original)
    policy.actor_state=states['actor']
    policy.qf_state=states['critic']
    policy.target_actor_state=states['target_actor']
    before = hashlib.sha256(fs.to_bytes(states)).hexdigest()
    rngs = [np.asarray(x).copy() for x in (policy.key,policy.noise_key,model.key)]
    a = model.cfg.alg.actor
    assert a.num_policy_samples==64 and a.proposals_per_policy_sample==1
    assert a.distillation_loss=='direct_gmm_nll' and a.source_q_eval=='mean'
    assert a.proposal_sampling_mode=='exact' and a.density_beta==1 and a.density_correction
    assert a.log_std_min==-5 and a.log_std_max==a.initial_log_std==-1
    from optiq_dime.semi_implicit import ConditionalGaussianProposal,actor_components
    from optiq_dime.distillation import direct_gmm_nll
    gradient = jax.jit(jax.grad(lambda mu,ls,u,w: direct_gmm_nll(mu,ls,u,w)[0],argnums=(0,1)))
    batch = len(observations)
    raw = dict(observations=observations,replay_indices=np.asarray(indices))
    records = {m:[] for m in (64,256,1024,4096)}
    started=time.monotonic()

    def cloud(mu,logs,key,count):
        proposal=ConditionalGaussianProposal(mu,logs,float(a.proposal_std))
        actions,u,component_ids=proposal.sample(key,count//64,'exact')
        logq=proposal.log_prob(u)
        obs=np.repeat(observations,count,axis=0)
        online,_=evaluate_q(policy,obs,np.asarray(actions).reshape(-1,8))
        q=online.mean(-1).reshape(batch,count)
        weights=jax.nn.softmax(jnp.asarray(q)/float(a.temperature)-logq,axis=-1)
        np.testing.assert_allclose(np.asarray(weights).sum(-1),1,rtol=1e-6,atol=1e-6)
        grads=gradient(mu,logs,jax.lax.stop_gradient(u),jax.lax.stop_gradient(weights))
        grad=np.concatenate([np.asarray(g).reshape(batch,-1) for g in grads],axis=-1)
        assert np.isfinite(grad).all() and np.isfinite(q).all() and np.isfinite(logq).all()
        w=np.asarray(weights);act=np.asarray(actions)
        return dict(actions=act,logq=np.asarray(logq),q=q,weights=w,gradient=grad,
                    action_mean=(w[...,None]*act).sum(1),q_mean=(w*q).sum(1),
                    ess=1/np.square(w).sum(1),max_weight=w.max(1),component_ids=np.asarray(component_ids))

    for rep in range(8):
        seed=923510+rep
        _,latent_key,proposal_key,_=jax.random.split(jax.random.PRNGKey(seed),4)
        zkey,_=jax.random.split(latent_key)
        mu,logs=actor_components(policy.actor_state,jnp.asarray(observations),zkey,64)
        reference=cloud(mu,logs,jax.random.fold_in(proposal_key,4097),4096)
        for count in records:
            key=proposal_key if count==64 else jax.random.fold_in(proposal_key,count)
            measured=cloud(mu,logs,key,count)
            if rep==0 and count==64:
                from types import SimpleNamespace
                existing,_=teacher_probe(SimpleNamespace(model=model),observations,seed)
                np.testing.assert_array_equal(existing['mu'],mu)
                np.testing.assert_array_equal(existing['candidates'],measured['actions'])
                np.testing.assert_allclose(existing['weights'],measured['weights'],rtol=2e-5,atol=2e-6)
            norm=np.linalg.norm(measured['gradient'],axis=1)
            refnorm=np.linalg.norm(reference['gradient'],axis=1)
            assert np.all(norm>1e-12) and np.all(refnorm>1e-12)
            cosine=(measured['gradient']*reference['gradient']).sum(1)/(norm*refnorm)
            row=dict(rep=rep,count=count,ess=measured['ess'].tolist(),max_weight=measured['max_weight'].tolist(),
                output_gradient_cosine=cosine.tolist(),
                output_gradient_relative_error=(np.linalg.norm(measured['gradient']-reference['gradient'],axis=1)/refnorm).tolist(),
                weighted_action_l2_error=np.linalg.norm(measured['action_mean']-reference['action_mean'],axis=1).tolist(),
                weighted_q_absolute_error=np.abs(measured['q_mean']-reference['q_mean']).tolist(),
                source_q_std=measured['q'].std(1).tolist())
            records[count].append(row)
            prefix=f'rep{rep}_M{count}_'
            for k,v in measured.items():raw[prefix+k]=v
        for k,v in reference.items():raw[f'rep{rep}_reference_'+k]=v
        write(args.output/'progress.json',dict(completed_repeats=rep+1,repeats=8,seconds=time.monotonic()-started))
    actual=dict(actor=policy.actor_state,critic=policy.qf_state,target_actor=policy.target_actor_state)
    assert hashlib.sha256(fs.to_bytes(actual)).hexdigest()==before
    for x,wanted in zip((policy.key,policy.noise_key,model.key),rngs):np.testing.assert_array_equal(x,wanted)
    assert sha(checkpoint)==proof['sha256']
    np.savez_compressed(args.output/'teacher-mc.npz',**raw)
    summary={}
    for count,rows in records.items():
        summary[count]={}
        for key in ('ess','max_weight','output_gradient_cosine','output_gradient_relative_error',
                    'weighted_action_l2_error','weighted_q_absolute_error','source_q_std'):
            x=np.asarray([row[key] for row in rows])
            summary[count][key]=dict(mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))
    write(args.output/'result.json',dict(completed=True,task=cfg['task'],training_source=cfg['source_commit'],
        reporting_source=reporting_sha,checkpoint_sha256=proof['sha256'],checkpoint_step=payload['step'],
        temperature=float(a.temperature),teacher_floor=float(a.proposal_std),states=selection,repeats=8,
        candidates=[64,256,1024,4096],student_components=64,reference_candidates=4096,
        comparison='Same states and same64 latent components within each repeat; independent candidate clouds',
        summary=summary,records=records,existing_M64_probe_matched=True,model_optimizer_rng_checkpoint_unchanged=True,
        training_performed=False,rollouts_performed=False,seconds=time.monotonic()-started,
        limitations=['4096 is a finite Monte Carlo reference, not an exact target or true Q.',
            'Gradients are with respect to conditional means/log sigmas, not full network parameters.',
            'Candidate actions are not trajectory routes; these statistics cannot establish successful multimodality.',
            'State anchors select actual nearest replay observations; requested coordinates may not be visited.',
            'Single seed and eight finite latent clouds; no new sampling or training setting has been launched.'],
        raw_sha256=sha(args.output/'teacher-mc.npz')))
    print(json.dumps(dict(completed=True,task=cfg['task'],summary=summary)),flush=True)


if __name__=='__main__':
    main()
