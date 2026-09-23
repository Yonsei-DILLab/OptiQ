"""Reproducible per-run launcher. Full actor/Adam/RNG checkpoints, no critic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time
import flax.serialization
import jax
import numpy as np
from .core import Experiment
from .evaluate import evaluate,score_diagnostic

def atomic_json(path,data):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    condition=cfg['conditions'][args.index%len(cfg['conditions'])];seed=cfg['seeds'][args.index//len(cfg['conditions'])]
    method,L=condition['method'],condition['L'];name=f'{method}_L{L}_s{seed}'
    out=args.root/'runtime/runs'/name;out.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.root/'SOURCE_MANIFEST.json').read_text())
    if (out/'COMPLETE.json').exists():
        print(f'Already complete: {name}',flush=True);return
    if (out/'RUN.json').exists() and not args.resume:
        raise RuntimeError('Existing run: explicit --resume is required')
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required; refusing login-node/CPU training')
    exp=Experiment(cfg,method,L,seed);stopping=[]
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,lambda n,f:stopping.append(n))
    metadata=dict(name=name,condition=condition,seed=seed,config=cfg,source_commit=source['commit'],
        upstream=source['upstream'],hostname=platform.node(),job_id=os.environ.get('SLURM_JOB_ID'),
        array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
        devices=[str(x) for x in jax.devices()],jax=jax.__version__,started=time.time(),
        initial_parameter_sha256=hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest())
    if args.resume:
        previous=json.loads((out/'RUN.json').read_text())
        if previous['source_commit']!=source['commit']:raise RuntimeError('Changed implementation cannot resume silently')
        exp.restore(out/'checkpoint.msgpack');metadata=previous
        with (out/'resume.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),job=os.environ.get('SLURM_JOB_ID')))+'\n')
    else:atomic_json(out/'RUN.json',metadata)
    def save():
        tmp=out/'checkpoint.tmp';exp.save(tmp);tmp.replace(out/'checkpoint.msgpack')
    try:
        start_step=int(exp.state.step);compile_start=time.perf_counter()
        exp.advance_fn.lower(exp.state,exp.key,50).compile()
        atomic_json(out/'COMPILE.json',dict(seconds=time.perf_counter()-compile_start))
        training_seconds=0.;timing_path=out/'training.jsonl'
        if args.resume and timing_path.exists():
            training_seconds=json.loads(timing_path.read_text().splitlines()[-1])['training_seconds']
        for step in range(start_step,cfg['steps']+1,50):
            atomic_json(out/'STATUS.json',dict(phase='evaluation' if step in cfg['eval_steps'] else 'training',step=step,time=time.time()))
            if step in cfg['eval_steps'] and not (out/f'metrics_{step:05d}.json').exists():
                metric=evaluate(exp,out,step);metric['training_seconds']=training_seconds
                atomic_json(out/f'metrics_{step:05d}.json',metric)
                print(json.dumps(metric),flush=True)
            if step in cfg['diagnostic_steps'] and not (out/f'score_{step:05d}.json').exists():score_diagnostic(exp,out,step)
            if step%500==0:save()
            if stopping:
                save();atomic_json(out/'STATUS.json',dict(phase='paused',step=step,signals=stopping,time=time.time()));return
            if step==cfg['steps']:break
            began=time.perf_counter();info=exp.advance(50);elapsed=time.perf_counter()-began;training_seconds+=elapsed
            if not all(np.isfinite(x) for x in info.values()):raise FloatingPointError(str(info))
            info.update(step=step+50,block_seconds=elapsed,training_seconds=training_seconds)
            with timing_path.open('a') as f:f.write(json.dumps(info)+'\n')
            if (step+50)%500==0:print(json.dumps(info),flush=True)
        atomic_json(out/'COMPLETE.json',dict(step=int(exp.state.step),finished=time.time(),training_seconds=training_seconds,source_commit=source['commit']))
        atomic_json(out/'STATUS.json',dict(phase='complete',step=int(exp.state.step),time=time.time()))
    except Exception as e:
        save();atomic_json(out/'STATUS.json',dict(phase='failed',step=int(exp.state.step),error=repr(e),time=time.time()));raise

if __name__=='__main__':main()
