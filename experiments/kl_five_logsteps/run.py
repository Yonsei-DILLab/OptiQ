"""Replay only the first 1K updates; never modify the ongoing 100K campaign."""
import argparse, hashlib, json, os, platform, signal, time
from pathlib import Path
import flax.serialization
import jax
import numpy as np
from ..kl_five_progress.run import checked, config, read, verify
from ..kl_diverse_targets_1d.run import write
from ..kl_six_highL_1d.evaluate import evaluate

STEPS = [0, 10, 100, 1000]
def state_bytes(exp):return flax.serialization.to_bytes({'state':exp.state,'key':exp.key})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();verify(a.root)
    assert jax.default_backend()=='gpu' and not jax.config.jax_threefry_partitionable
    task=read(a.root/'data/TASKS.json')[a.index];cfg=config(task)
    parent=a.parent/'runtime/replay'/task['case']/f'{task["method"]}_s{task["seed"]}'
    previous=read(parent/'RUN.json')
    assert previous['initial_parameter_sha256']==task['initial_parameter_sha256']
    assert previous['L']==task['L'] and previous['seed']==task['seed']
    exp=checked(task)(cfg,task['method'],task['L'],task['seed'])
    out=a.root/'runtime/early'/task['case']/f'{task["method"]}_s{task["seed"]}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    commit=read(a.root/'SOURCE_MANIFEST.json')['commit']
    metadata=dict(source_commit=commit,config=cfg,method=task['method'],L=task['L'],seed=task['seed'],
        index=a.index,stop_after=1000,initial_parameter_sha256=task['initial_parameter_sha256'],
        parent=str(parent),parent_commit=previous['source_commit'],hostname=platform.node(),
        job=os.environ.get('SLURM_JOB_ID'),array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),started=time.time(),
        note='Independently replayed early segment; not recovered intermediate checkpoints.')
    if (out/'RUN.json').exists():
        old=read(out/'RUN.json');assert old['source_commit']==commit and old['config']==cfg
        if (out/'checkpoint.msgpack').exists():exp.restore(out/'checkpoint.msgpack')
        with (out/'RESUME.jsonl').open('a') as f:f.write(json.dumps(metadata)+'\n')
    else:write(out/'RUN.json',metadata)
    stop=[]
    for sig in [signal.SIGTERM,signal.SIGUSR1,signal.SIGINT]:signal.signal(sig,lambda s,f:stop.append(s))
    def save():
        tmp=out/'checkpoint.tmp';exp.save(tmp);tmp.replace(out/'checkpoint.msgpack')
    try:
        while True:
            step=int(exp.state.step)
            write(out/'STATUS.json',dict(step=step,phase='training',time=time.time()))
            if step in STEPS and not (out/f'metrics_{step:06d}.json').exists():
                before=state_bytes(exp);exp.save(out/f'checkpoint_{step:06d}.msgpack')
                metric=evaluate(exp,out,step)
                assert before==state_bytes(exp),'Evaluation mutated training state'
                metric.update(replayed_early_segment=True,evaluation_preserves_state=True)
                write(out/f'metrics_{step:06d}.json',metric)
                print(json.dumps(dict(case=task['case'],method=task['method'],seed=task['seed'],step=step,
                    TV=metric['histogram_TV'],samples=metric['sample_count'])),flush=True)
            if step%100==0 or step in STEPS:save()
            if stop:
                save();write(out/'STATUS.json',dict(step=step,phase='paused',time=time.time()));return
            if step==1000:break
            endpoint=next(s for s in STEPS if s>step)
            # Original scan blocks: reverse 5, forward 100. Split only the first
            # forward block into 10+90 to observe update 10; N/M/batch unchanged.
            block=5 if task['method']=='reverse' else 100
            count=min(block,endpoint-step);t=time.perf_counter();info=exp.advance(count)
            assert all(np.isfinite(x) for x in info.values()),info
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=int(exp.state.step),seconds=time.perf_counter()-t,**info))+'\n')
        save();write(out/'COMPLETE.json',dict(step=1000,source_commit=commit,time=time.time()))
        write(out/'STATUS.json',dict(step=1000,phase='complete',time=time.time()))
    except Exception as e:
        save();write(out/'STATUS.json',dict(step=int(exp.state.step),phase='failed',error=repr(e),time=time.time()));raise
if __name__=='__main__':main()
