import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize._numdiff import approx_derivative
from .core import *
from .evaluate import density_cdf,evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',action='store_true');a=ap.parse_args()
    if a.gpu:assert any(d.platform=='gpu' for d in jax.devices())
    allcfg=configurations(load_config());checks=[]
    assert len(allcfg)==75
    for case in load_config()['cases']:
        cfg=next(c for c in allcfg if c['id']==case['id'] and c['k']==15 and c['variance']=='learned' and c['weights']=='learned')
        x,m=quadrature(cfg,2049);xx,mm=quadrature(cfg,8193);p=initialize(cfg)[0]
        v,g=jax.value_and_grad(lambda p:loss(p,jnp.array(x),jnp.array(m),cfg))(p)
        vv,gg=jax.value_and_grad(lambda p:loss(p,jnp.array(xx),jnp.array(mm),cfg))(p)
        assert abs(v-vv)<1e-7 and np.max(np.abs(g-gg))<1e-7
        checks.append(dict(case=case['id'],quad_loss_diff=float(abs(v-vv)),quad_grad_diff=float(np.max(np.abs(g-gg)))))
    cfg=allcfg[0];exp=Trainer(cfg);before=np.asarray(exp.state[0]);exp.state,vals=exp.advance(exp.state,200);jax.block_until_ready(exp.state)
    assert np.array_equal(before[:,1:],np.asarray(exp.state[0])[:,1:])
    assert np.max(np.asarray(vals[-1])-np.asarray(vals[0]))<0
    # Test mean/variance/weight gradients for the full truncated normalized mixture.
    cfg=next(c for c in allcfg if c['k']==3 and c['weights']=='learned');p=initialize(cfg)[0];x,m=quadrature(cfg,2049)
    fn=lambda v:float(loss(jnp.asarray(v.reshape(3,3)),jnp.array(x),jnp.array(m),cfg))
    fd=approx_derivative(fn,np.asarray(p).ravel(),method='3-point').ravel()
    ad=np.asarray(jax.grad(lambda p:loss(p,jnp.array(x),jnp.array(m),cfg))(p)).ravel()
    assert np.max(np.abs(fd-ad))<1e-6
    # Known representable equal-weight target at K=3 must have KL=0 and zero gradient.
    cfg=dict(allcfg[0]);p=np.stack([cfg['target_centers'],np.full(3,np.log(.5)),np.zeros(3)])
    pdf,cdf=density_cdf(p,cfg,x);assert abs(cdf[-1]-cdf[0]-1)<1e-12
    assert np.max(np.abs(pdf-reference(cfg,x)[0]))<1e-12
    assert np.max(np.abs(jax.grad(lambda p:loss(p,jnp.array(x),jnp.array(m),cfg))(jnp.asarray(p))))<1e-9
    # Unbounded paper gradient equals E[responsibility*(mu-x)/sigma^2].
    cfg=allcfg[-2];x,m=quadrature(cfg,2049);p=initialize(cfg)[0];mu=np.asarray(p[0])
    resp=np.asarray(jax.nn.softmax(-.5*((x[:,None]-mu)/.5)**2,axis=-1));analytic=np.sum(m[:,None]*resp*(mu-x[:,None])/.25,axis=0)
    actual=np.asarray(jax.grad(lambda p:loss(p,jnp.array(x),jnp.array(m),cfg))(p))[0]
    assert np.max(np.abs(analytic-actual))<1e-10
    # Exercise final evaluation, learned weights, Hessian and serialization paths.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for cfg0 in [allcfg[10],allcfg[-2]]:
            smoke=dict(cfg0,steps=2,final_eval_samples=512)
            folder=Path(d)/smoke['stage'];folder.mkdir()
            met=evaluate(np.asarray(initialize(smoke)[0]),smoke,folder,2,0)
            assert np.isfinite(met['forward_KL']) and met['sample_count']==512
    # Benchmark representative largest K without consuming any production trial.
    cfg=allcfg[9];exp=Trainer(cfg);exp.state,_=exp.advance(exp.state,100);jax.block_until_ready(exp.state)
    t=time.time();exp.state,_=exp.advance(exp.state,1000);jax.block_until_ready(exp.state);dt=time.time()-t
    result=dict(passed=True,devices=str(jax.devices()),checks=checks,gradient_fd_max_error=float(np.max(np.abs(fd-ad))),
                benchmark_seconds_per_update=dt/1000,benchmark_includes_first_1000_step_compile=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
