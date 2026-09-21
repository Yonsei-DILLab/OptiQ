"""Batch means, one Adam update, microbatch equivalence and RNG resume."""
from .core import *
from experiments.gmm_mode_gradient.validate import check
from flax import serialization
from pathlib import Path
import json,time


def main():
    results=[]
    state,key=initialize(0);nk,keys=batch_keys(key)
    for n,m in [(16,16),(64,64)]:
        for method in METHODS:
            one=group_fn(n,m,method)
            # Explicit Python reference, all groups from SAME pre-update params.
            groups=[one(state.params,k) for k in keys]
            mean=jax.tree_util.tree_map(lambda *x:sum(x)/BATCH_SIZE,*[x[0] for x in groups])
            met=sum(x[1] for x in groups)/BATCH_SIZE
            for micro in (1,4,32):
                got,metric=aggregate_fn(n,m,method,micro)(state.params,keys)
                check(mean,got,atol=3e-6,rtol=3e-4);check(met,metric)
            new,newkey,metric=engine(n,m,method)['step'](state,key)
            expected=state.apply_gradients(grads=mean)
            check(new,expected,atol=3e-6,rtol=3e-4);check(newkey,nk,atol=0,rtol=0)
            assert int(new.step)==int(state.step)+1
            assert np.isfinite(np.asarray(metric)).all()
            saved=serialization.to_bytes(new);restored=serialization.from_bytes(new,saved)
            check(engine(n,m,method)['step'](new,newkey),engine(n,m,method)['step'](restored,newkey),atol=0,rtol=0)
            if method=='baseline':
                t,_=base(n,m)['teacher'](state.params,OBS,keys[0],QARG)
                check(groups[0][0],jax.grad(base(n,m)['loss'])(state.params,t),atol=0,rtol=0)
            results.append(dict(n=n,m=m,method=method,mean_gradient=True,one_adam_step=True,microbatch_equivalent=True,resume_bitwise=True))
    from .diagnostics import probe
    arrays,summary=probe(state,key,16,16,0,'mode_confidence')
    assert arrays['batch_H'].shape==(32,16,3)
    check(arrays['batch_H'].sum(-1),np.ones((32,16)))
    assert np.isfinite(arrays['routed_gradient_norm']).all()
    # Mode gradients should sum to the baseline batch gradient.
    from .diagnostics import batch_mode_grads_fn
    gs,_=batch_mode_grads_fn(16,16)(state.params,keys)
    check(jax.tree_util.tree_map(lambda x:x.sum(0),gs),aggregate_fn(16,16,'baseline')(state.params,keys)[0])
    output=dict(passed=True,batch_size=BATCH_SIZE,devices=[str(x) for x in jax.devices()],tests=results,time=time.time())
    print(json.dumps(output),flush=True)
    path=bootstrap.ROOT/'runtime'/'VALIDATION.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
