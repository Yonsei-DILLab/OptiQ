"""Meaningful parity, score/gradient and reference checks before GPU screening."""
import argparse,json,time,tempfile
from pathlib import Path
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from .core import Experiment,dense_score
from .evaluate import reference
from ..kl_forward_wide_1d.core import Experiment as OldForward
from ..kl_reverse_wide_1d.core import Experiment as OldReverse
from ..kl_forward_wide_1d.box_gaussian import sample_box,mixture_log_prob


def flat(t):return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(t)])
def close(a,b,tol=3e-5):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert error<tol*max(1,float(np.max(np.abs(b)))),error
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--gpu',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());checks={}
    for method,old,L in [('forward',OldForward,0),('reverse',OldReverse,1024)]:
        c=dict(cfg,n=8,m=11,batch=2,hidden_dims=[16,16]);a=Experiment(c,method,L,0);b=old(c,method,L,0)
        assert flax.serialization.to_bytes(a.state)==flax.serialization.to_bytes(b.state)
        key=jax.random.PRNGKey(31)
        ga=jax.grad(lambda q:a.group_loss(q,key)[0])(a.state.params);gb=jax.grad(lambda q:b.group_loss(q,key)[0])(b.state.params)
        checks[method+'_parity_gradient']=close(flat(ga),flat(gb),1e-6)
    c=dict(c,**cfg['cases'][0]);e=Experiment(c,'reverse',1024,0);key=jax.random.PRNGKey(701)
    sk,ik,ek,dk=jax.random.split(key,4);z=jax.random.normal(sk,(8,1));idx=jax.random.randint(ik,(11,),0,8)
    aux=jnp.concatenate([jax.random.normal(jax.random.fold_in(dk,i),(256,1)) for i in range(4)])
    aa=jnp.linspace(-9.7,9.7,31)[:,None];mu,ls=e.density_components(e.state.params,aux)
    s,lp,_=e.density_score(e.state.params,aa,dk,1024);s0,lp0=dense_score(aa,mu,ls)
    checks['score_dense']=close(s,s0);checks['logdensity_dense']=close(lp,lp0)
    def direct(params):
        cm,cl=e.components(params,z);act=sample_box(ek,cm[idx],cl[idx])
        dm,dl=e.density_components(jax.tree_util.tree_map(jax.lax.stop_gradient,params),aux)
        return (mixture_log_prob(act[None],dm[None],dl[None])[0]-e.log_f(act)).mean()
    g=jax.grad(lambda p:e.group_loss(p,key)[0])(e.state.params)
    checks['reverse_surrogate_gradient']=close(flat(g),flat(jax.grad(direct)(e.state.params)),1e-4)
    for case in cfg['cases']:
        cc=dict(cfg,**case);ee=Experiment(cc,'forward',0,0)
        checks[case['id']+'_target_score']=close(ee.target_score(aa),jax.grad(lambda a:ee.log_f(a).sum())(aa))
        _,cdf=reference(cc,[-10,10]);assert abs(cdf[0])<1e-12 and abs(cdf[-1]-1)<1e-12
        assert np.min(np.diff(case['target_centers']))>=5*case['target_width']
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'checkpoint';e.advance(2);e.save(path);e.advance(2);expected=flat(e.state.params)
        e.restore(path);e.samples(128);e.advance(2);checks['resume_rng']=close(flat(e.state.params),expected,1e-7)
    timing=[]
    if args.gpu:
        assert jax.default_backend()=='gpu'
        jax.clear_caches()
        for method,L in [('forward',0),('reverse',1024)]:
            ee=Experiment(dict(cfg,**cfg['cases'][0]),method,L,0)
            ee.advance(100);t=time.perf_counter();info=ee.advance(100);seconds=time.perf_counter()-t
            assert all(np.isfinite(x) for x in info.values())
            timing.append(dict(method=method,seconds_per_update=seconds/100,info=info))
            jax.clear_caches()
    report=dict(passed=True,backend=jax.default_backend(),checks=checks,benchmark=timing)
    if args.out:args.out.parent.mkdir(exist_ok=True,parents=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
