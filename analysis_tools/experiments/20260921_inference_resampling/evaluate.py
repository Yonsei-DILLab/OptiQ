"""Post-hoc mu-only resampling: frozen actor AND live critic, no learner updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    job = manifest['jobs'][args.index]
    source = Path(job['source'])
    sys.path.insert(0, str(source/'analysis_tools/studies/20260918_nonstationary_nd/v5'))
    sys.path.insert(0, str(source/'analysis_tools/experiments/20260920_truncated_mll'))
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from omegaconf import OmegaConf
    from optiq_dime.policy import OptiQPolicy
    from optiq_dime.box_gaussian import mixture_log_prob

    cfg = OmegaConf.load(job['config'])
    assert cfg.dacer.enabled and cfg.alg.actor.temperature == .25
    assert cfg.alg.actor.get('latent_prior','normal') == 'normal'
    assert cfg.alg.actor.source_q_eval == 'mean'
    assert list(cfg.alg.actor.hidden_dims) == [256,256]
    assert cfg.alg.critic.n_atoms == 1 and cfg.alg.critic.n_critics == 2
    assert not cfg.alg.optimizer.bn and not cfg.alg.critic.dropout_rate
    assert {k:sha(job[k]) for k in ['config','actor','critic']} == job['input_sha256']
    env = gym.make(cfg.env_name)
    policy = OptiQPolicy(env.observation_space, env.action_space, cfg)
    policy.build(jax.random.PRNGKey(int(cfg.seed)), lambda _: cfg.alg.optimizer.lr_actor,
                 cfg.alg.optimizer.lr_critic)
    policy.actor_state = serialization.from_bytes(policy.actor_state, Path(job['actor']).read_bytes())
    policy.qf_state = serialization.from_bytes(policy.qf_state, Path(job['critic']).read_bytes())
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    dim = env.action_space.shape[0]
    astate, qstate = policy.actor_state, policy.qf_state
    before = hashlib.sha256(serialization.to_bytes((astate,qstate))).hexdigest()
    modes = manifest.get('evaluation_modes',['mu_one','mu_q64','mu_kde_is64'])
    assert all(m in ['mu_one','mu_q64','mu_kde_is64','mu_best64'] for m in modes)
    episodes = manifest['episodes']
    seed_base = manifest['seed_base']
    reset_seeds = [seed_base+i for i in range(episodes)]
    policy_seeds = [seed_base+10000+i for i in range(episodes)]
    ref_n = manifest['density_reference_samples']

    def log_weights(q, log_density):
        return jax.nn.log_softmax((q-q.max(axis=-1,keepdims=True))/.25-log_density, axis=-1)

    def make_step(mode):
        k = 1 if mode == 'mu_one' else 64
        @jax.jit
        def step(astate,qstate,obs,keys):
            splits = jax.vmap(lambda key:jax.random.split(key))(keys)
            new_keys, noise_keys = splits[:,0],splits[:,1]
            latent_keys = jax.vmap(lambda key:jax.random.split(key)[0])(noise_keys)
            z = jax.vmap(lambda key:jax.random.normal(key,(k,dim)))(latent_keys)
            repeated = jnp.broadcast_to(obs[:,None],(len(obs),k,obs.shape[-1]))
            mu,_ = astate.apply_fn({'params':astate.params},repeated.reshape(-1,obs.shape[-1]),z.reshape(-1,dim))
            mu = mu.reshape(len(obs),k,dim)
            qs = qstate.apply_fn({'params':qstate.params,'batch_stats':qstate.batch_stats},
                                repeated.reshape(-1,obs.shape[-1]),mu.reshape(-1,dim),train=False)
            q = qs[...,0].mean(axis=0).reshape(len(obs),k)
            log_q = jnp.zeros_like(q)
            bandwidth_mean = jnp.zeros(len(obs))
            if mode == 'mu_kde_is64':
                reference_keys = jax.vmap(lambda key:jax.random.fold_in(key,3001))(noise_keys)
                rz = jax.vmap(lambda key:jax.random.normal(key,(ref_n,dim)))(reference_keys)
                ro = jnp.broadcast_to(obs[:,None],(len(obs),ref_n,obs.shape[-1]))
                centers,_ = astate.apply_fn({'params':astate.params},ro.reshape(-1,obs.shape[-1]),rz.reshape(-1,dim))
                centers = centers.reshape(len(obs),ref_n,dim)
                # Independent pilot mu draws; diagonal Scott bandwidth, normalized action units.
                # Boundary-normalized Gaussian KDE is an estimate, NOT the actor's sigma density.
                h = jnp.maximum(centers.std(axis=1,ddof=1)*ref_n**(-1/(dim+4)),.001)
                ls = jnp.broadcast_to(jnp.log(h)[:,None],centers.shape)
                log_q = mixture_log_prob(mu,centers,ls)
                bandwidth_mean = h.mean(-1)
            lw = log_weights(q,log_q)
            if mode == 'mu_one':
                indices = jnp.zeros(len(obs),dtype=jnp.int32)
            elif mode == 'mu_best64':
                indices = jnp.argmax(q,axis=-1)
            else:
                select_keys = jax.vmap(lambda key:jax.random.fold_in(key,3002))(noise_keys)
                indices = jax.vmap(jax.random.categorical)(select_keys,lw)
            weights = jnp.exp(lw)
            if mode == 'mu_best64':
                weights = jax.nn.one_hot(indices,k)
            selected = mu[jnp.arange(len(obs)),indices]
            stats = jnp.stack([1/(weights**2).sum(-1),weights.max(-1),
                               q[jnp.arange(len(obs)),indices]-q.mean(-1),bandwidth_mean],axis=-1)
            return selected,new_keys,stats
        return step

    steps = {mode:make_step(mode) for mode in modes}
    obs,_ = env.reset(seed=reset_seeds[0])
    obs = jnp.asarray(np.broadcast_to(obs,(episodes,len(obs))),dtype=jnp.float32)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in policy_seeds])
    # The baseline must be exactly the existing mu-only evaluator at identical keys.
    ours,_,_ = make_step('mu_one')(astate,qstate,obs,keys)
    nk = jax.vmap(lambda key:jax.random.split(key)[1])(keys)
    original = jax.vmap(lambda o,key:policy.sample_action(astate,o[None],key,
                        deterministic=False,sample_conditional_noise=False)[0])(obs,nk)
    np.testing.assert_allclose(ours,original,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(np.exp(log_weights(jnp.zeros((1,4)),jnp.zeros((1,4)))),.25)
    q=jnp.array([[1.,2.,3.]])
    np.testing.assert_allclose(log_weights(q,jnp.zeros_like(q)),log_weights(q+100,jnp.zeros_like(q)),atol=1e-6)
    # All evaluation modes must be invariant to changing ONLY sigma-head parameters.
    altered=dict(astate.params)
    altered['log_std']=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*123,altered['log_std'])
    altered=astate.replace(params=altered)
    for mode,fn in steps.items():
        action,_,stats=fn(astate,qstate,obs,keys)
        alt,_,altstats=fn(altered,qstate,obs,keys)
        np.testing.assert_array_equal(action,alt)
        np.testing.assert_array_equal(stats,altstats)
        assert np.isfinite(action).all() and np.isfinite(stats).all()
        assert ((np.asarray(action)>=-1)&(np.asarray(action)<=1)).all()
        if mode == 'mu_best64':
            assert (np.asarray(stats[:,2])>=-1e-5).all()
            np.testing.assert_array_equal(stats[:,:2],np.ones((episodes,2)))
    record=dict(job=job,protocol={k:v for k,v in manifest.items() if k!='jobs'},
                evaluation_commit=manifest['evaluation_commit'],preflight_passed=True,
                packages=dict(jax=jax.__version__,gymnasium=gym.__version__),
                actor_step=int(astate.step),critic_step=int(qstate.step),
                modes={},complete=False)
    out=Path(manifest['output'])/(job['name']+('.preflight.json' if args.preflight else '.json'))
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(name=job['name'],preflight='passed')),flush=True)
        env.close();return
    assert not out.exists(),f'Refuse to overwrite {out}'
    env.close()
    for mode,fn in steps.items():
        ve=gym.vector.SyncVectorEnv([lambda:gym.make(cfg.env_name) for _ in range(episodes)])
        observations,_=ve.reset(seed=reset_seeds)
        keys=jnp.stack([jax.random.PRNGKey(s) for s in policy_seeds])
        returns=np.zeros(episodes);lengths=np.zeros(episodes,dtype=int);done=np.zeros(episodes,dtype=bool)
        totals=np.zeros((episodes,4));infer_seconds=0.;started=time.monotonic()
        while not done.all():
            t=time.perf_counter()
            actions,keys,stats=fn(astate,qstate,jnp.asarray(observations),keys)
            actions=np.asarray(actions);stats=np.asarray(stats)
            infer_seconds+=time.perf_counter()-t
            assert np.isfinite(actions).all() and np.isfinite(stats).all()
            observations,rewards,terminated,truncated,_=ve.step(actions)
            active=~done
            returns[active]+=rewards[active];lengths[active]+=1;totals[active]+=stats[active]
            done |= terminated | truncated
            assert lengths.max()<=1000
        ve.close()
        result=dict(returns=returns.tolist(),lengths=lengths.tolist(),mean_return=float(returns.mean()),
                    episode_sd=float(returns.std(ddof=1)),reset_seeds=reset_seeds,policy_seeds=policy_seeds,
                    mean_ess=float((totals[:,0]/lengths).mean()),
                    mean_max_weight=float((totals[:,1]/lengths).mean()),
                    mean_selected_q_gain=float((totals[:,2]/lengths).mean()),
                    mean_kde_bandwidth=float((totals[:,3]/lengths).mean()),
                    wall_seconds=time.monotonic()-started,inference_seconds=infer_seconds,
                    inference_ms_per_vector_step=1000*infer_seconds/int(lengths.max()))
        record['modes'][mode]=result
        out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(name=job['name'],mode=mode,mean_return=result['mean_return'],ess=result['mean_ess'])),flush=True)
    after=hashlib.sha256(serialization.to_bytes((astate,qstate))).hexdigest()
    assert before==after
    assert {k:sha(job[k]) for k in ['config','actor','critic']}==job['input_sha256']
    record.update(complete=True,checkpoint_state_unchanged=True)
    out.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
