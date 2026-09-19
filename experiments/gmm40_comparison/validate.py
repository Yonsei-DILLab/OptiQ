"""Numerical gates and representative timing before the queue can start."""
import argparse,itertools,json,os,time
from pathlib import Path
import flax.serialization
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from benchmarks.gmm40 import sampler
from optiq_dime.transport import sinkhorn
from .engine import cfg_for,initialize,engine,monge_indices,component_logp,draw
from .v5_distribution import conditional_ot_nll,ConditionalGaussianProposal
from .metrics import reference,evaluate,plot_snapshot,component_stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    assert jax.config.jax_default_matmul_precision=='highest'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3' and len(jax.devices())==1 and jax.default_backend()=='gpu'
    out=a.root/'validation';out.mkdir(parents=True,exist_ok=True)
    report=dict(commit=manifest['commit'],gpu=str(jax.devices()[0]),checks={},timings={})
    # Legacy extracted path agrees with the original actor update, not just loss algebra.
    cfg=cfg_for(dict(name='validation',method='legacy',N=16,M=64),7)
    actor,key,target=initialize(cfg);_,oracle,_=sampler.initialize(7,cfg['hidden_dims'],coordinate_scale=1.)
    old_cfg=sampler.default_config();old_cfg.update(batch_size=1,num_policy_samples=16,proposals_per_policy_sample=4,
        proposal_std=8.,density_beta=1.,temperature=1.,sinkhorn_epsilon=.01,sinkhorn_iterations=300,
        unbounded_actions=True,coordinate_scale=1.,latent_sampling='grid')
    new,nkey,v=engine('legacy',16,64)['step'](actor,key,target)
    expected,ev,ekey,_=sampler.update(actor,oracle,key,old_cfg)
    error=max(float(jnp.max(jnp.abs(x-y))) for x,y in zip(jax.tree_util.tree_leaves(new.params),jax.tree_util.tree_leaves(expected.params)))
    np.testing.assert_allclose(v,ev,rtol=1e-5,atol=1e-5);assert error<3e-5
    np.testing.assert_array_equal(nkey,ekey);report['checks']['legacy_update_max_parameter_error']=error
    # Exact 2D assignment vs brute force; N!=M weighted quantization is explicit.
    rng=np.random.default_rng(2);x=rng.normal(size=(5,2));b=rng.normal(size=(5,2));w=np.ones(5)/5
    ids=monge_indices(x,b,w,.5,np.arange(5));assert len(set(ids))==5
    cost=((x-b[ids])**2).sum();opt=min(((x-b[list(p)])**2).sum() for p in itertools.permutations(range(5)))
    np.testing.assert_allclose(cost,opt);report['checks']['monge_brute_force_cost']=cost
    b2=rng.normal(size=(19,2));w2=rng.uniform(size=19);w2/=w2.sum()
    ids2=monge_indices(x,b2,w2,.37,rng.permutation(19));assert ids2.shape==(5,) and ids2.min()>=0 and ids2.max()<19
    report['checks']['rectangular_weighted_teacher_supported']=True
    # OT and GMM losses tested against explicit double sums, plus finite gradients.
    mu=jnp.asarray(rng.normal(size=(5,2)));ls=jnp.asarray(rng.normal(size=(5,2))*.1);u=jnp.asarray(b2)
    wj=jnp.asarray(w2);R=jax.nn.softmax(jnp.asarray(rng.normal(size=(5,19))),axis=1)
    ell=component_logp(u,mu,ls)
    ot=conditional_ot_nll(mu[None],ls[None],u[None],R[None]);np.testing.assert_allclose(ot,-(R*ell).sum()/5,rtol=1e-6)
    gm=-(wj*(jax.scipy.special.logsumexp(ell,axis=0)-jnp.log(5))).sum()
    explicit=-np.sum(w2*np.log(np.exp(np.asarray(ell)).mean(0)));np.testing.assert_allclose(gm,explicit,rtol=1e-6)
    assert np.isfinite(jax.grad(lambda m:-(wj*(jax.scipy.special.logsumexp(component_logp(u,m,ls),0)-jnp.log(5))).sum())(mu)).all()
    report['checks']['losses_explicit_and_finite_gradients']=True
    # Direct GMM must never call the Sinkhorn solver.
    import importlib
    module=importlib.import_module('experiments.gmm40_comparison.engine');original=module.sinkhorn
    def forbidden(*args,**kwargs):raise AssertionError('Direct GMM called Sinkhorn')
    module.sinkhorn=forbidden
    cg=cfg_for(dict(name='gmmtest',method='gmm',N=8,M=32),0);ag,kg,tg=initialize(cg)
    ag,kg,loss=engine('gmm',8,32)['step'](ag,kg,tg);jax.block_until_ready(ag.params)
    module.sinkhorn=original;report['checks']['direct_gmm_no_ot']=True
    # GMM and v5 have identical actors, teachers, and initialization for each size.
    co=cfg_for(dict(name='ottest',method='v5_ot',N=8,M=32),0);ao,ko,to=initialize(co)
    ag,kg,tg=initialize(cg)
    gt,_=engine('gmm',8,32)['teacher'](ag,kg,tg);ot,_=engine('v5_ot',8,32)['teacher'](ao,ko,to)
    for name in ['z','b','u','w','q','log_q']:np.testing.assert_array_equal(gt[name],ot[name])
    report['checks']['paired_initialization_and_teacher_equal']=True
    # Checkpoint actor/Adam/RNG restore gives the identical next update.
    encoded=flax.serialization.msgpack_serialize(dict(actor=flax.serialization.to_state_dict(ag),key=np.asarray(kg)))
    decoded=flax.serialization.msgpack_restore(encoded);restored=flax.serialization.from_state_dict(ag,decoded['actor'])
    lhs,lk,lv=engine('gmm',8,32)['step'](ag,kg,tg)
    rhs,rk,rv=engine('gmm',8,32)['step'](restored,decoded['key'],tg)
    for x,y in zip(jax.tree_util.tree_leaves(lhs),jax.tree_util.tree_leaves(rhs)):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(lk,rk);report['checks']['checkpoint_next_update_exact']=True
    td,ds=engine('gmm',8,32)['diagnostics'](ag,kg,tg);td=jax.device_get(td)
    sl=np.asarray(draw(ag,jax.random.PRNGKey(999),32768,'gmm'));loc=np.asarray(tg['locs']);sc=np.asarray(tg['scales'])
    rr=reference(loc,sc);mm,st=evaluate(sl,rr,loc,sc)
    plot_snapshot(out,0,sl,rr,loc,st,td,'gmm')
    report['checks']['sample_and_assignment_figures_rendered']=True
    # Sinkhorn only promises finite-iteration approximate marginals; toy residual gate.
    C=jnp.asarray(rng.uniform(0,.1,(1,5,19)),jnp.float32)
    P=np.asarray(sinkhorn(C,wj[None],.1,100))[0]
    err=max(abs(P.sum(0)-w2).max(),abs(P.sum(1)-.2).max());assert err<2e-6
    report['checks']['well_conditioned_sinkhorn_max_marginal_error']=float(err)
    # Timings include an explicit compilation warmup, followed by 100 actual updates.
    plan=json.loads((Path(__file__).parent/'plan.json').read_text())
    for condition in plan['conditions']:
        cfg=cfg_for(condition,0);actor,key,target=initialize(cfg);algo=engine(cfg['method'],cfg['N'],cfg['M'])
        t0=time.monotonic();actor,key,loss=algo['block'](actor,key,target,100);jax.block_until_ready(actor.params)
        compile_time=time.monotonic()-t0;t0=time.monotonic()
        actor,key,loss=algo['block'](actor,key,target,100);jax.block_until_ready(actor.params)
        seconds=(time.monotonic()-t0)/100
        assert np.isfinite(float(loss));assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(jax.device_get(actor.params)))
        samples=np.asarray(draw(actor,jax.random.PRNGKey(31),1024,cfg['method']));assert np.isfinite(samples).all()
        record=dict(seconds_per_update=seconds,compile_and_100_updates_seconds=compile_time,loss=float(loss))
        report['timings'][condition['name']]=record;print(json.dumps(dict(condition=condition['name'],**record)),flush=True)
    locs=np.asarray(target['locs']);scales=np.asarray(target['scales']);ref=reference(locs,scales)
    floor,_=evaluate(reference(locs,scales,seed=20260920),ref,locs,scales)
    report['reference_sampling_floor']=floor;report['passed']=True
    (out/'VALIDATION_PASSED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
