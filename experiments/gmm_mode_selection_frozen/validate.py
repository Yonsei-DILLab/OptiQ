"""Validate original references, output-gradient routing, batch averaging and resume."""
from .core import *
from flax import serialization
from pathlib import Path
import json,time,hashlib

def check(a,b,atol=3e-6,rtol=3e-4):
    assert jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)

def main():
    results=[];here=Path(__file__).parent
    fixture=json.loads((here/'legacy_reference_fixture.json').read_text())
    assert hashlib.sha256((here/'legacy_problems.py').read_bytes()).hexdigest()==fixture['source_sha256']
    state,key=initialize(0);nk,keys=batch_keys(key)
    for name in landscape.LANDSCAPES:
        f=functions(name,16,16);ref=f['ref']
        for field,value in fixture['references'][name].items():np.testing.assert_allclose(ref[field],value,atol=2e-7,rtol=2e-6)
        assert abs(sum(ref['bin_mass'])-1)<1e-12
        x=np.linspace(-.999,.999,1001)
        np.testing.assert_allclose(landscape.q_jax(jnp.asarray(x[:,None]),name),.25*landscape.log_energy(x,name),atol=5e-6,rtol=2e-6)
        t,_=f['teacher'](state.params,keys[0]);old,_=base(16,16)['teacher'](state.params,OBS,keys[0],QARG)
        for field in ('z','b','u','log_q'):check(t[field],old[field],atol=0,rtol=0)
        d=f['assignment'](state.params,t);check(d['joint'].sum(0),t['w'][0]);check(d['H'].sum(1),jnp.ones(16))
        mu,ls=heads(state.params,t['z'][0]);k=f['k']
        def output_terms(mm,ll):
            _,ell=direct_gmm_nll(mm[None],ll[None],t['u'],t['w'])
            logmix=jax.scipy.special.logsumexp(ell[0],axis=0)-jnp.log(16)
            return jax.nn.one_hot(d['labels'],k).T@(-t['w'][0]*logmix)
        out=jax.jacrev(output_terms,argnums=(0,1))(mu,ls)
        for method in METHODS:
            g,value,dd,coef=f['gradient'](state.params,t,method)
            gate=jnp.ones((k,16)) if method=='baseline' else jax.nn.one_hot(d['mode'],k).T
            selected=tuple((v*gate[:,:,None]).sum(0) for v in out)
            _,vjp=jax.vjp(lambda p:heads(p,t['z'][0]),state.params);check(g,vjp(selected)[0])
            if method=='baseline':check(g,jax.grad(base(16,16)['loss'])(state.params,t),atol=0,rtol=0)
            else:
                check(coef.sum(),(d['alpha']*d['confidence']).sum())
                assert np.all(np.asarray(coef)[np.asarray(d['mode'][:,None]!=d['labels'][None,:])]==0)
                gw=jax.grad(lambda w:sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(f['gradient'](state.params,dict(t,w=w),'mode_only')[0])))(t['w'])
                assert np.all(np.asarray(gw)==0)
        results.append(dict(landscape=name,original_reference=True,proposal_unchanged=True,baseline_original=True,mode_vjp=True,no_confidence_weight=True))
    name='needle3'
    for method in METHODS:
        groups=[group_fn(name,16,16,method)(state.params,k) for k in keys]
        mean=jax.tree_util.tree_map(lambda *xs:sum(xs)/BATCH_SIZE,*[v[0] for v in groups]);met=sum(v[1] for v in groups)/BATCH_SIZE
        for micro in (1,8,128):
            got,metric=aggregate_fn(name,16,16,method,micro=micro)(state.params,keys);check(mean,got);check(met,metric)
        new,newkey,metric=engine(name,16,16,method)['step'](state,key)
        check(new,state.apply_gradients(grads=mean));check(newkey,nk,atol=0,rtol=0);assert int(new.step)==1
        restored=serialization.from_bytes(new,serialization.to_bytes(new))
        check(engine(name,16,16,method)['step'](new,newkey),engine(name,16,16,method)['step'](restored,newkey),atol=0,rtol=0)
        results.append(dict(method=method,batch128_mean=True,microbatch_equivalent=True,one_adam_step=True,resume_bitwise=True))
    from .diagnostics import probe,batch_mode_grads_fn
    arrays,summary=probe(state,key,'comb6',16,16,0,'mode_only')
    assert arrays['batch_H'].shape==(128,16,6) and arrays['joint'].shape==(16,16)
    gs,_=batch_mode_grads_fn('comb6',16,16)(state.params,keys)
    check(jax.tree_util.tree_map(lambda x:x.sum(0),gs),aggregate_fn('comb6',16,16,'baseline')(state.params,keys)[0])
    assert np.isfinite(arrays['gradient_gram']).all()
    output=dict(passed=True,batch_size=BATCH_SIZE,devices=[str(x) for x in jax.devices()],tests=results,time=time.time())
    print(json.dumps(output),flush=True)
    path=bootstrap.ROOT/'runtime'/'VALIDATION.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
