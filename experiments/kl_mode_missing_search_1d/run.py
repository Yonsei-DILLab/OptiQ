"""Independent case/seed jobs. Screening then separately initialized confirmation."""
import argparse,hashlib,json,os,platform,signal,time
from pathlib import Path
import flax.serialization
import jax
import numpy as np
from .core import Experiment
from .evaluate import evaluate


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def train(root,cfg,case,seed,method,steps,L,stage):
    out=root/'runtime'/stage/case['id']/f'{method}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    cfg=dict(cfg,**case,steps=steps,density_chunk=4096 if L>1024 else 256,
             train_block=5 if L>1024 else 100,checkpoint_interval=100 if L>1024 else 1000)
    source=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    exp=Experiment(cfg,method,L,seed)
    metadata=dict(config=cfg,method=method,L=L,seed=seed,stage=stage,source_commit=source['commit'],
                  initial_parameter_sha256=hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest(),
                  hostname=platform.node(),job=os.environ.get('SLURM_JOB_ID'),started=time.time())
    if (out/'RUN.json').exists():
        previous=json.loads((out/'RUN.json').read_text());assert previous['source_commit']==source['commit']
        assert previous['L']==L and previous['config']==cfg
        if (out/'checkpoint.msgpack').exists():exp.restore(out/'checkpoint.msgpack')
        with (out/'RESUME.jsonl').open('a') as f:f.write(json.dumps(metadata)+'\n')
    else:write(out/'RUN.json',metadata)
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,lambda s,f:stop.append(s))
    def save():
        tmp=out/'checkpoint.tmp';exp.save(tmp);tmp.replace(out/'checkpoint.msgpack')
    try:
        started=time.perf_counter();exp.advance_fn.lower(exp.state,exp.key,cfg['train_block']).compile()
        write(out/'COMPILE.json',dict(seconds=time.perf_counter()-started))
        for step in range(int(exp.state.step),steps+1,cfg['train_block']):
            write(out/'STATUS.json',dict(step=step,phase='training',time=time.time()))
            if step in cfg['eval_steps'] and not (out/f'metrics_{step:06d}.json').exists():
                metric=evaluate(exp,out,step);write(out/f'metrics_{step:06d}.json',metric)
                print(json.dumps(dict(case=case['id'],method=method,seed=seed,**metric)),flush=True)
            if step%cfg['checkpoint_interval']==0:save()
            if stop:
                save();write(out/'STATUS.json',dict(step=step,phase='paused',time=time.time()));raise SystemExit(0)
            if step==steps:break
            t=time.perf_counter();info=exp.advance(cfg['train_block']);info.update(step=step+cfg['train_block'],seconds=time.perf_counter()-t)
            if not all(np.isfinite(v) for v in info.values()):raise FloatingPointError(str(info))
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
        save();write(out/'COMPLETE.json',dict(step=int(exp.state.step),source_commit=source['commit'],time=time.time()))
        write(out/'STATUS.json',dict(step=int(exp.state.step),phase='complete',time=time.time()))
    except Exception as e:
        save();write(out/'STATUS.json',dict(step=int(exp.state.step),phase='failed',error=repr(e),time=time.time()));raise
    jax.clear_caches()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--stage',choices=['screen','replicate','confirm'],default='screen');p.add_argument('--selected-case',type=int)
    args=p.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    assert jax.default_backend()=='gpu','Never train on login node'
    if args.stage=='screen':
        case=cfg['cases'][args.index//2];seed=args.index%2
        for method in ['forward','reverse']:train(args.root,cfg,case,seed,method,10000,1024 if method=='reverse' else 0,'screen')
    elif args.stage=='replicate':
        case=cfg['cases'][args.selected_case];seed=2+args.index
        for method in ['forward','reverse']:train(args.root,cfg,case,seed,method,10000,1024 if method=='reverse' else 0,'replicate')
    else:
        case=cfg['cases'][args.selected_case];seed=args.index//2;method=['forward','reverse'][args.index%2]
        train(args.root,cfg,case,seed,method,100000,1048576 if method=='reverse' else 0,'confirm')

if __name__=='__main__':main()
