"""Behavioral and integration checks before launching the five-seed ablation."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,jax.scipy as jsp
from flax import serialization
from .shared import actor_state,config,dummy_critic,ot_update
from .categorical_worker import configure_categorical
from .problems import make_problem
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.transport import TruncatedGaussianKDE,sinkhorn

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);a=ap.parse_args();root=Path(a.campaign)
    protocol=json.loads((root/'protocol.json').read_text());base=Path(protocol['baseline'])
    spec=importlib.util.spec_from_file_location('optiq_dime.original_argmax_algorithm',base/'code/optiq_dime/algorithm.py')
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    cfg=config().alg.actor
    def call(fn,state,critic,obs,key,mode):
        return fn(state,critic,obs,key,jnp.array([0.]),int(cfg.num_policy_samples),int(cfg.proposals_per_policy_sample),cfg.proposal_sampling_mode,float(cfg.proposal_std),float(cfg.proposal_clip),bool(cfg.include_anchor),bool(cfg.density_correction),float(cfg.density_beta),bool(cfg.adaptive_density_beta),float(cfg.minimum_source_ess),int(cfg.density_beta_grid_size),float(cfg.temperature),float(cfg.sinkhorn_epsilon),int(cfg.sinkhorn_iterations),cfg.source_q_eval,mode)
    rows=[]
    for case in ['unimodal','modes2d_8','separable_8']:
        p=make_problem(case);state=actor_state(p.dim,0);state=serialization.from_bytes(state,(base/'runs'/f'frozen_{case}_coverage_seed0/actor_0.msgpack').read_bytes())
        critic=dummy_critic(lambda obs,a:p.q(a,jnp,jsp.special.logsumexp));obs=jnp.zeros((32,1));key=jax.random.PRNGKey(1000)
        ref,rl,rk,rm=call(original.OptiQDIME.update_actor,state,critic,obs,key,'argmax')
        same,sl,sk,sm=call(OptiQDIME.update_actor,state,critic,obs,key,'argmax')
        assert float(rl)==float(sl) and np.array_equal(rk,sk)
        assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(ref.params),jax.tree.leaves(same.params)))
        result,loss,outkey,metrics=call(OptiQDIME.update_actor,state,critic,obs,key,'categorical')
        configure_categorical(0)
        wrapped,wrapped_loss,wrapped_key,wrapped_metrics=ot_update(state,critic,obs,key)
        assert float(wrapped_loss)==float(loss), 'Actual frozen-Q wrapper is not using categorical'
        assert np.array_equal(wrapped_key,outkey)
        assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(wrapped.params),jax.tree.leaves(result.params)))
        assert not all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(wrapped.params),jax.tree.leaves(same.params))), 'Categorical wrapper unexpectedly matches argmax'
        assert np.isfinite(float(loss)) and all(np.isfinite(x).all() for x in jax.tree.leaves(result.params))
        assert np.array_equal(outkey,rk), 'RNG stream changed'
        # Independently reconstruct this batch to check its actual target loss.
        returnkey,lk,pk,dk=jax.random.split(key,4);n=int(cfg.num_policy_samples);m=int(cfg.proposals_per_policy_sample);batch=len(obs)
        z=jax.random.normal(lk,(batch,n,p.dim),dtype=obs.dtype)
        raw=state.apply_fn({'params':state.params},jnp.zeros((batch*n,1)),z.reshape(-1,p.dim)).reshape(batch,n,p.dim)
        acts=jnp.clip(raw,-1,1);kde=TruncatedGaussianKDE.from_centers(acts,float(cfg.proposal_std),float(cfg.proposal_clip));candidates=kde.sample_stratified(pk,m,bool(cfg.include_anchor)).reshape(batch,n*m,p.dim)
        q=p.q(candidates,jnp,jsp.special.logsumexp);w=jax.nn.softmax(q/.25-kde.log_prob(candidates),axis=-1)
        cost=jnp.sum((acts[:,:,None,:]-candidates[:,None,:,:])**2,axis=-1);cost/=cost.mean((-2,-1),keepdims=True)+1e-8
        t=sinkhorn(cost,w,float(cfg.sinkhorn_epsilon),int(cfg.sinkhorn_iterations));r=t/jnp.maximum(t.sum(-1,keepdims=True),1e-20)
        indices=jax.random.categorical(jax.random.fold_in(returnkey,0x434154),jnp.where(r>0,jnp.log(r),-jnp.inf),axis=-1)
        target=jax.vmap(lambda c,i:c[i])(candidates,indices);manual=jnp.mean(jnp.sum((raw-target)**2,axis=-1))
        # Fused JIT and eager evaluation can differ slightly in floating point.
        assert np.isclose(float(manual),float(loss),rtol=2e-4,atol=2e-5),(case,float(manual),float(loss))
        row=dict(case=case,argmax_unchanged=True,categorical_loss=float(loss),actual_frozen_wrapper_loss=float(wrapped_loss),wrapper_matches_categorical=True,wrapper_differs_from_argmax=True,manual_loss=float(manual),loss_difference=abs(float(manual)-float(loss)),rng_unchanged=True)
        rows.append(row);print(json.dumps(row),flush=True)
    # Nonuniform categorical distribution: catches accidentally retaining argmax.
    probabilities=jnp.array([.1,.2,.7]);draw=jax.random.categorical(jax.random.PRNGKey(19),jnp.log(probabilities),shape=(100000,))
    freq=np.bincount(np.asarray(draw),minlength=3)/len(draw);assert abs(freq-np.asarray(probabilities)).max()<.006,freq
    (root/'smoke_passed.json').write_text(json.dumps(dict(rows=rows,categorical_frequencies=freq.tolist(),passed=True),indent=2))
    print('SMOKE_PASSED',flush=True)

if __name__=='__main__':main()
