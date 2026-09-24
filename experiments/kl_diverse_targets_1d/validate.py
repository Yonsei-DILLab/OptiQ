"""Target math, inherited-update parity, resume and GPU timing verification."""
import argparse,json,time,tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization
from scipy.integrate import quad
from .core import Experiment
from .target import reference,geometry
from ..kl_mode_missing_search_1d.core import Experiment as Prior


def flat(t):return np.concatenate([np.asarray(a).ravel() for a in jax.tree_util.tree_leaves(t)])
def close(a,b,tol=1e-5):
    delta=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert delta < tol*max(1,float(np.max(np.abs(b)))),delta
    return delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text());checks={}
    a=jnp.linspace(-9.7,9.7,101)[:,None]
    small=dict(cfg,n=8,m=11,batch=2,hidden_dims=[16,16])
    for case in cfg['cases']:
        c=dict(small,**case);e=Experiment(c,'reverse',1024,0)
        checks[case['id']+'_score']=close(e.target_score(a),jax.grad(lambda x:e.log_f(x).sum())(a),3e-5)
        vals=reference(c,[-10,10])[1];assert abs(vals[0])<1e-12 and abs(vals[1]-1)<1e-12
        integral=quad(lambda x:float(reference(c,x)[0]),-10,10,epsabs=1e-10,points=case['target_centers'])[0]
        assert abs(integral-1)<1e-8;checks[case['id']+'_normalization_error']=abs(integral-1)
        geom=geometry(c);checks[case['id']+'_mode_count']=len(geom['peaks'])
        assert np.all(geom['target_core']>0) and np.all(geom['target_basin']>.02)
    reference_case=cfg['cases'][0]
    for method,L in [('forward',0),('reverse',1024)]:
        c=dict(small,**reference_case,target_width=.5);e=Experiment(c,method,L,0);old=Prior(c,method,L,0)
        assert flax.serialization.to_bytes(e.state)==flax.serialization.to_bytes(old.state)
        key=jax.random.PRNGKey(61)
        ga=jax.grad(lambda p:e.group_loss(p,key)[0])(e.state.params)
        gb=jax.grad(lambda p:old.group_loss(p,key)[0])(old.state.params)
        checks[method+'_prior_gradient']=close(flat(ga),flat(gb),3e-5)
        checks[method+'_prior_target']=close(e.log_f(a),old.log_f(a),1e-5)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'c';e.advance(2);e.save(path);e.advance(2);expected=flat(e.state.params)
        e.restore(path);e.samples(128);e.advance(2);checks['resume_rng']=close(flat(e.state.params),expected,1e-7)
    timing=[]
    if args.gpu:
        assert jax.default_backend()=='gpu'
        for method,L in [('forward',0),('reverse',1024)]:
            jax.clear_caches();e=Experiment(dict(cfg,**reference_case),method,L,0)
            e.advance(100);t=time.perf_counter();info=e.advance(100)
            assert all(np.isfinite(v) for v in info.values())
            timing.append(dict(method=method,seconds_per_update=(time.perf_counter()-t)/100))
    result=dict(passed=True,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],checks=checks,timing=timing)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
