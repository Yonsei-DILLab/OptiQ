"""Numerical checks; --gpu also benchmarks full requested shapes for 5 conditions."""
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from .core import Experiment,target_score,log_f,q_value,dense_score
from .box_gaussian import sample_box,component_log_prob,mixture_log_prob
from .evaluate import cdf,Z,target_logf

def flat(tree):return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)])
def close(a,b,tolerance=3e-5):
    a,b=np.asarray(a),np.asarray(b)
    error=float(np.max(np.abs(a-b)));scale=max(1.,float(np.max(np.abs(b))))
    assert error<tolerance*scale,(error,scale)
    return error

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    report=dict(backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],checks={})
    if args.gpu:assert jax.default_backend()=='gpu'
    checks=report['checks'];upstream=json.loads(Path(__file__).with_name('UPSTREAM.json').read_text())
    for name,record in upstream['vendored'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==record['sha256']
    checks['upstream_kernel_and_forward_loss_hashes']='match'
    checks['normalization']=close(quad(lambda x:np.exp(target_logf(x))/Z,-1,1)[0],1,1e-9)
    checks['cdf_endpoints']=close(cdf(np.array([-1.,1.])),[0,1],1e-10)
    a=jnp.linspace(-.97,.97,31)[:,None]
    checks['target_gradient']=close(target_score(a),jax.grad(lambda x:log_f(x).sum())(a))
    small=dict(cfg,n=8,m=11,batch=2,hidden_dims=[16,16],density_chunk=8)
    e=Experiment(small,'reverse',32,0);key=jax.random.PRNGKey(701)
    source_key,idx_key,eps_key,aux_key=jax.random.split(key,4)
    z=jax.random.normal(source_key,(8,1));indices=jax.random.randint(idx_key,(11,),0,8)
    aux_z=jnp.concatenate([jax.random.normal(jax.random.fold_in(aux_key,i),(8,1)) for i in range(4)])
    mu,ls=e.components(e.state.params,aux_z)
    score,logp,_=e.density_score(e.state.params,a,aux_key,32)
    expected,expected_lp=dense_score(a,mu,ls)
    checks['chunk_score']=close(score,expected);checks['chunk_log_density']=close(logp,expected_lp)
    checks['score_vs_action_autograd']=close(expected,jax.grad(lambda x:mixture_log_prob(x[None],mu[None],ls[None]).sum())(a))
    def direct_path(params):
        cm,cl=e.components(params,z);acts=sample_box(eps_key,cm[indices],cl[indices])
        pm,pl=e.components(jax.tree_util.tree_map(jax.lax.stop_gradient,params),aux_z)
        return (mixture_log_prob(acts[None],pm[None],pl[None])[0]-q_value(acts)/cfg['temperature']).mean()
    grad=jax.grad(lambda p:e.group_loss(p,key)[0])(e.state.params)
    expected_grad=jax.grad(direct_path)(e.state.params)
    checks['reverse_surrogate_vs_autograd']=close(flat(grad),flat(expected_grad),1e-4)
    assert np.linalg.norm(flat(grad))>1e-5
    checks['auxiliary_parameter_gradient_zero']=close(flat(jax.grad(lambda p:e.density_score(p,a,aux_key,32)[1].sum())(e.state.params)),0.)
    # Forward equivalence to explicit stopped SNIS-weighted marginal log density.
    f=Experiment(small,'forward',0,0)
    def forward_reference(params):
        cm,cl=f.components(params,z);pm,pl=jax.lax.stop_gradient(cm),jax.lax.stop_gradient(cl)
        acts=jax.lax.stop_gradient(sample_box(eps_key,pm[indices],pl[indices]))
        prop=mixture_log_prob(acts[None],pm[None],pl[None])[0]
        w=jax.lax.stop_gradient(jax.nn.softmax(q_value(acts)/cfg['temperature']-prop))
        return -(w*mixture_log_prob(acts[None],cm[None],cl[None])[0]).sum()
    checks['forward_loss_gradient']=close(flat(jax.grad(lambda p:f.group_loss(p,key)[0])(f.state.params)),flat(jax.grad(forward_reference)(f.state.params)))
    initial=flax.serialization.to_bytes(f.state.params)
    for cond in cfg['conditions']:
        other=Experiment(small,cond['method'],cond['L'],0)
        assert flax.serialization.to_bytes(other.state.params)==initial
    checks['paired_initialization']='all five identical'
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state';f.advance(2);f.save(path);info=f.advance(2);expected_params=flat(f.state.params);expected_key=np.asarray(f.key)
        f.restore(path);f.samples(128);f.advance(2)
        checks['full_resume_and_eval_rng_isolation']=close(flat(f.state.params),expected_params,1e-7)
        assert np.array_equal(np.asarray(f.key),expected_key)
    if args.gpu:
        benchmarks=[]
        for cond in cfg['conditions']:
            e=Experiment(cfg,cond['method'],cond['L'],0)
            start=time.perf_counter();e.advance_fn.lower(e.state,e.key,10).compile();compile_seconds=time.perf_counter()-start
            start=time.perf_counter();metrics=e.advance(10);seconds=time.perf_counter()-start
            assert all(np.isfinite(x) for x in metrics.values());assert metrics['gradient_norm']>0
            assert metrics['sigma_min']>=np.exp(-5)-1e-6 and metrics['sigma_max']<=np.exp(-1)+1e-6
            benchmarks.append(dict(**cond,compile_seconds=compile_seconds,seconds_per_update=seconds/10,metrics=metrics))
            # Release compiled full-shape executable before next smoke condition.
            del e;jax.clear_caches()
        report['full_shape_benchmarks']=benchmarks
    report['passed']=True
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
