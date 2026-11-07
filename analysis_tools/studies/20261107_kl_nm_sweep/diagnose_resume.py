"""Distinguish checkpoint corruption, RNG changes, and GPU numerical variation."""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--diagnostic-commit',required=True);a=p.parse_args();root=a.root.resolve()
sys.path.insert(0,str(root/'source'))
from common import tasks,write,append,now
from worker import gpu_info

aid='resume_diagnostic_'+str(int(time.time()));out=root/'runtime/diagnostic'/aid
attempt=out/'attempts'/aid;attempt.mkdir(parents=True)
task=tasks('pilot')[1]
meta={k:v for k,v in task.items() if k!='config'}
meta.update(id=aid,phase='diagnostic',attempt_id=aid,source_commit=a.diagnostic_commit,
            numerical_source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gpu=gpu_info(0),gpu_index=0,gpu_count=1,start_utc=now(),start_epoch=time.time())
write(attempt/'BEGIN.json',meta)
started=time.monotonic();stop=threading.Event()
def telemetry():
    while not stop.is_set():
        elapsed=time.monotonic()-started
        append(attempt/'telemetry.jsonl',dict(elapsed_seconds=elapsed,**gpu_info(0)))
        write(attempt/'CURRENT.json',dict(meta,allocated_gpu_hours=elapsed/3600))
        stop.wait(5)
thread=threading.Thread(target=telemetry,daemon=True);thread.start()
rc=1
try:
    import numpy as np
    import jax
    import flax.serialization
    from experiments.kl_six_highL_1d.core import implementation
    exp=implementation(task['config'])(task['config'],'reverse',1048576,0)
    assert jax.default_backend()=='gpu'
    exp.advance(5)
    original_state,original_key=exp.state,exp.key
    exp.save(out/'before.msgpack')
    def snapshot():return jax.tree_util.tree_map(lambda x:np.array(x,copy=True),(exp.state,exp.key))
    before=snapshot()
    exp.advance(5);continued=snapshot()
    exp.state,exp.key=original_state,original_key
    exp.advance(5);repeated_device=snapshot()
    exp.restore(out/'before.msgpack');restored=snapshot()
    restored_step_weak=getattr(exp.state.step,'weak_type',None)
    exp.advance(5);continued_restore=snapshot()
    def difference(left,right):
        lefts,_=jax.tree_util.tree_flatten_with_path(left);rights,_=jax.tree_util.tree_flatten_with_path(right)
        changes=[]
        for (path,x),(_,y) in zip(lefts,rights):
            x=np.asarray(x);y=np.asarray(y)
            if not np.array_equal(x,y):
                diff=np.abs(x.astype(float)-y.astype(float))
                changes.append(dict(path=str(path),shape=x.shape,dtype=str(x.dtype),
                     max_abs=float(diff.max()),l2_relative=float(np.linalg.norm(diff.ravel())/max(np.linalg.norm(x.ravel()),1e-30)),
                     allclose_1e6=bool(np.allclose(x,y,rtol=1e-6,atol=1e-7))))
        return changes
    report=dict(checkpoint_contents=difference(before,restored),
                repeated_device=difference(continued,repeated_device),
                continued_vs_restored=difference(continued,continued_restore),
                original_step_weak=getattr(original_state.step,'weak_type',None),restored_step_weak=restored_step_weak,
                rng_equal=bool(np.array_equal(continued[1],continued_restore[1])))
    write(out/'DIAGNOSTIC.json',report);print(json.dumps(report,indent=2),flush=True)
    rc=0
finally:
    stop.set();thread.join(timeout=20)
    elapsed=time.monotonic()-started
    write(attempt/'END.json',dict(meta,end_utc=now(),end_epoch=time.time(),exit_code=rc,
          allocated_gpu_hours=elapsed/3600,elapsed_seconds=elapsed,synchronized_training_gpu_hours=None,
          timing_status='diagnostic end-to-end; compilation included'))
    write(root/'allocations'/f'{aid}.json',dict(id=aid,start_epoch=meta['start_epoch'],end_epoch=time.time(),
          gpu_count=1,gpu_index=0,source_commit=a.diagnostic_commit,allocated_gpu_hours=elapsed/3600))
