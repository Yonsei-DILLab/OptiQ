"""Unchanged optimizer kernels; additional evaluations do not consume training RNG."""
import argparse, hashlib, importlib.metadata, json, os, platform, tempfile, time
from pathlib import Path
import flax.serialization
import jax
import numpy as np
from ..kl_diverse_targets_1d.run import train, write
from ..kl_six_highL_1d.core import implementation
from ..kl_six_highL_1d.evaluate import evaluate

def read(p): return json.loads(p.read_text())
def digest(exp): return hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest()
def config(task):
    return dict(task['config'],allow_large_L=True,eval_samples=2**20,final_eval_samples=2**20,
        final_sample_chunk=16384,histogram_bins=512,
        eval_steps=sorted(set(range(0,100001,10000))|{1000,25000,75000}))

def verify(root):
    for name,expected in read(root/'SOURCE_MANIFEST.json')['files'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,name
    for name,expected in read(root/'data/INPUT_MANIFEST.json').items():
        assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==expected,name

def checked(task):
    def build(cfg,method,L,seed):
        exp=implementation(cfg)(cfg,method,L,seed)
        assert digest(exp)==task['initial_parameter_sha256'], 'Original initialization mismatch'
        return exp
    return build

def check_reference(exp,ref):
    actions,mu,sigma=exp.samples(ref['probe_draw_count'])
    result={}
    for name,values in [('actions',actions),('mu',mu),('sigma',sigma)]:
        delta=np.asarray(values).ravel()[:128]-np.asarray(ref[name])
        result[name+'_max_abs_difference']=float(np.abs(delta).max())
        result[name+'_rmse']=float(np.sqrt(np.mean(delta**2)))
    result['bitwise_probe_match']=all(result[k]==0 for k in result if k.endswith('difference'))
    result['original_samples_sha256']=ref['sha256']
    return result

def diagnostics(root,task):
    refs=read(root/'data/REFERENCE.json')
    def callback(exp,out,step):
        # Serialize the entire optimizer and RNG state to check evaluation purity.
        before=flax.serialization.to_bytes({'state':exp.state,'key':exp.key})
        if step%10000==0:
            tmp=out/f'checkpoint_{step:06d}.tmp';exp.save(tmp)
            tmp.replace(out/f'checkpoint_{step:06d}.msgpack')
        key=f'{task["index"]}:{step}'
        compare=check_reference(exp,refs[key]) if key in refs else None
        metric=evaluate(exp,out,step)
        after=flax.serialization.to_bytes({'state':exp.state,'key':exp.key})
        assert before==after, 'Evaluation changed optimizer or training RNG'
        if compare is not None:
            metric['original_fixed_probe_comparison']=compare
            write(out/f'ORIGINAL_COMPARISON_{step:06d}.json',compare)
        metric['reconstructed_trajectory']=True
        return metric
    return callback

def preflight(root,pool):
    tasks=read(root/'data/TASKS.json');records=[]
    for index in [0,4,20,24]:
        task=tasks[index];cfg=config(task)
        exp=checked(task)(cfg,task['method'],task['L'],task['seed'])
        count=5 if task['L'] else 100
        t=time.perf_counter();first=exp.advance(count);comp=time.perf_counter()-t
        with tempfile.TemporaryDirectory(dir=root/'runtime') as tmp:
            cp=Path(tmp)/'checkpoint';exp.save(cp)
            t=time.perf_counter();info=exp.advance(count);elapsed=time.perf_counter()-t
            expected=flax.serialization.to_bytes({'state':exp.state,'key':exp.key})
            exp.restore(cp);key=np.asarray(exp.key).copy();exp.samples(16384)
            assert np.array_equal(key,np.asarray(exp.key))
            exp.advance(count)
            assert expected==flax.serialization.to_bytes({'state':exp.state,'key':exp.key})
            assert all(np.isfinite(v) for v in first.values()) and all(np.isfinite(v) for v in info.values())
            exp.restore(cp)
            if task['method']=='forward':
                metric=diagnostics(root,task)(exp,Path(tmp),int(exp.state.step))
                assert metric['sample_count']==2**20
        records.append(dict(case=task['case'],method=task['method'],L=task['L'],
            initial_hash_verified=True,resume_with_sampling_exact=True,
            seconds_per_update=elapsed/count,compile_and_first_block_seconds=comp))
        print(json.dumps(records[-1]),flush=True);del exp;jax.clear_caches()
    write(root/f'runtime/PREFLIGHT_{pool}.json',dict(passed=True,records=records,
        job=os.environ.get('SLURM_JOB_ID'),host=platform.node(),time=time.time(),
        versions={k:importlib.metadata.version(k) for k in ['jax','jaxlib','flax','optax','numpy']},
        threefry_partitionable=bool(jax.config.jax_threefry_partitionable)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index',type=int);p.add_argument('--preflight');a=p.parse_args()
    verify(a.root);assert jax.default_backend()=='gpu','Do not train on the login node'
    assert not jax.config.jax_threefry_partitionable
    (a.root/'runtime').mkdir(exist_ok=True)
    if a.preflight: preflight(a.root,a.preflight);return
    task=read(a.root/'data/TASKS.json')[a.index];cfg=config(task)
    folder=a.root/'runtime/replay'/task['case']/f'{task["method"]}_s{task["seed"]}'
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'ORIGIN.json',task)
    train(a.root,cfg,{'id':task['case']},task['seed'],task['method'],100000,task['L'],
        'replay',experiment_cls=checked(task),evaluate_fn=diagnostics(a.root,task))
if __name__=='__main__': main()
