"""Verify routing at the same outputs, not projection of summed shared gradients."""
from .core import *
from flax import serialization
from pathlib import Path
import json,time

def check(a,b,atol=3e-6,rtol=2e-4):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)

def main():
    results=[]
    for n,m in [(16,16),(64,4096),(2048,2048)]:
        state,key=initialize(0);f=base(n,m);t,k=f['teacher'](state.params,OBS,key,QARG)
        mu,ls=heads(state.params,t['z'][0]);d=assignment(state.params,t)
        check(d['H'].sum(1),jnp.ones(n));check(d['joint'].sum(0),t['w'][0])
        orig=jax.grad(f['loss'])(state.params,t)
        restored=gradients(state.params,t,'baseline')[0];check(orig,restored)
        for method in METHODS:
            g,_,coef=gradients(state.params,t,method)
            # Explicit mode-specific output gradients, then one network VJP.
            def terms(mm,ll):
                _,ell=direct_gmm_nll(mm[None],ll[None],t['u'],t['w'])
                lm=jax.scipy.special.logsumexp(ell[0],axis=0)-jnp.log(n)
                lab=jnp.digitize(t['b'][0,:,0],jnp.array([-.3,.3]))
                return jax.nn.one_hot(lab,3).T@(-t['w'][0]*lm)
            out=jax.jacrev(terms,argnums=(0,1))(mu,ls)
            gate=jnp.ones((3,n)) if method=='baseline' else jax.nn.one_hot(d['mode'],3).T
            if method=='mode_confidence':gate=gate*d['confidence'][None]
            selected=tuple((v*gate[:,:,None]).sum(0) for v in out)
            _,vjp=jax.vjp(lambda p:heads(p,t['z'][0]),state.params)
            expected=vjp(selected)[0];check(g,expected)
            new,metric=engine(n,m,method)['update'](state,t)
            assert np.isfinite(np.asarray(metric)).all()
            if method=='baseline':check(new,f['update'](state,t)[0],atol=0,rtol=0)
            else:
                # Finite differences of detached-coefficient surrogate (not original NLL).
                v=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.0001,state.params)
                pp=jax.tree_util.tree_map(lambda x,y:x+y,state.params,v)
                pm=jax.tree_util.tree_map(lambda x,y:x-y,state.params,v)
                fd=(surrogate(pp,t,coef,method)-surrogate(pm,t,coef,method))/2
                analytic=sum((x*y).sum() for x,y in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(v)))
                np.testing.assert_allclose(fd,analytic,atol=5e-6,rtol=.04)
            saved=serialization.to_bytes(new)
            restoredstate=serialization.from_bytes(new,saved)
            aa=engine(n,m,method)['step'](new,k);bb=engine(n,m,method)['step'](restoredstate,k)
            check(aa,bb,atol=0,rtol=0)
        a,ak,av=engine(n,m,'baseline')['block'](state,key,3)[:3]
        b,bk,_=block(n,m)(state,key,3);check(a,b,atol=0,rtol=0);check(ak,bk,atol=0,rtol=0)
        results.append(dict(n=n,m=m,mode_gradient_vjp=True,baseline_bitwise=True,resume_bitwise=True))
    # Teacher/assignment is detached, including confidence; absent mode gradient=0.
    state,key=initialize(3);t,_=base(16,16)['teacher'](state.params,OBS,key,QARG)
    for method in ('mode_only','mode_confidence'):
        fn=lambda tt: sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(gradients(state.params,tt,method)[0]))
        gw=jax.grad(lambda w:fn(dict(t,w=w)))(t['w']);assert np.all(np.asarray(gw)==0)
    from .diagnostics import probe
    arrays,summary=probe(state,key,16,16,3,'mode_confidence')
    assert np.isfinite(arrays['output_mode_gradients']).all()
    assert np.isfinite(arrays['routed_gradient_norm']).all()
    output=dict(passed=True,devices=[str(x) for x in jax.devices()],tests=results,time=time.time())
    print(json.dumps(output),flush=True)
    path=bootstrap.ROOT/'runtime'/'VALIDATION.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
