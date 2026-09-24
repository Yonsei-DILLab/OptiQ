import argparse,json,time,tempfile
from pathlib import Path
import jax
import numpy as np
import flax.serialization
from ..kl_diverse_targets_1d.core import Experiment as OldGMM
from ..kl_nongmm_targets_1d.core import Experiment as OldNG
from .core import implementation
from .evaluate import evaluate

def flat(t):return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(t)])
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());checks={};times=[]
    for case in cfg['cases']:
        c=dict(cfg,**case,n=8,m=11,batch=2,hidden_dims=[16,16],density_chunk=256)
        old_cls=OldNG if case['target_kind']=='nongmm' else OldGMM
        for method,L in [('forward',0),('reverse',1024)]:
            e=implementation(c)(c,method,L,0);old=old_cls(c,method,L,0)
            assert flax.serialization.to_bytes(e.state)==flax.serialization.to_bytes(old.state)
            key=jax.random.PRNGKey(77)
            g=flat(jax.grad(lambda p:e.group_loss(p,key)[0])(e.state.params));h=flat(jax.grad(lambda p:old.group_loss(p,key)[0])(old.state.params))
            diff=float(np.max(abs(g-h)));assert diff==0,diff;checks[case['id']+'_'+method+'_gradient_difference']=diff
    with tempfile.TemporaryDirectory() as tmp:
        e.advance(2);e.save(Path(tmp)/'c');e.advance(2);expected=flat(e.state.params)
        e.restore(Path(tmp)/'c');e.samples(128);e.advance(2);assert np.array_equal(flat(e.state.params),expected)
    if a.gpu:
        assert jax.default_backend()=='gpu'
        for case in cfg['cases'][:2]:
            jax.clear_caches();c=dict(cfg,**case);e=implementation(c)(c,'reverse',1048576,0)
            t=time.perf_counter();e.advance(5);compile_time=time.perf_counter()-t
            t=time.perf_counter();info=e.advance(10);elapsed=time.perf_counter()-t;assert all(np.isfinite(v) for v in info.values())
            times.append(dict(case=case['id'],L=1048576,seconds_per_update=elapsed/10,compile_and_first5_seconds=compile_time))
            with tempfile.TemporaryDirectory() as tmp:
                before=np.asarray(e.key).copy();metric=evaluate(e,Path(tmp),100000)
                assert metric['sample_count']==1048576 and np.array_equal(before,np.asarray(e.key));checks[case['id']+'_final_eval_samples']=metric['sample_count']
    result=dict(passed=True,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],checks=checks,timing=times,resume_rng_exact=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
