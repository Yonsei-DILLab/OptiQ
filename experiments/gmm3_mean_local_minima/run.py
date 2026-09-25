import argparse,json,time,os,signal
from pathlib import Path
import numpy as np
import jax
from .core import *
STOP=False

def stop(*_):
    global STOP
    STOP=True


def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    cfg=load_config();case=cases(cfg)[a.index];out=a.root/'runs'/case['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    manifest=json.loads((a.root/'SOURCE_MANIFEST.json').read_text());commit=manifest['commit'];t0=time.time()
    write(out/'CONFIG.json',dict(config=cfg,case=case,commit=commit,job=os.environ.get('SLURM_JOB_ID')))
    trainer=Trainer(case,cfg);step=0;checkpoint=out/'checkpoint.npz'
    if checkpoint.exists():
        q=np.load(checkpoint);assert str(q['commit'])==commit;trainer.mu=jnp.asarray(q['mu']);step=int(q['step'])
    def save():
        tmp=out/'checkpoint.tmp'
        with tmp.open('wb') as f:np.savez(f,mu=np.asarray(trainer.mu),step=step,commit=commit)
        tmp.replace(checkpoint)
    for s in [signal.SIGTERM,signal.SIGUSR1,signal.SIGINT]:signal.signal(s,stop)
    while True:
        if step%cfg['checkpoint_interval']==0 or STOP:
            val,grad=trainer.vg(trainer.mu);row=dict(step=step,time=time.time(),means=np.asarray(trainer.mu).tolist(),nll=np.asarray(val).tolist(),gradient_norm=np.linalg.norm(np.asarray(grad),axis=-1).tolist())
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'STATUS.json',row);save()
            print(case['id'],step,float(np.max(np.asarray(val))),flush=True)
        if step==cfg['steps'] or STOP:break
        trainer.mu=trainer.advance(trainer.mu,min(cfg['block'],cfg['steps']-step));jax.block_until_ready(trainer.mu)
        assert np.isfinite(np.asarray(trainer.mu)).all()
        step+=cfg['block']
    if step<cfg['steps']:return
    write(out/'COMPLETE.json',dict(step=step,elapsed=time.time()-t0,commit=commit,meta=trainer.meta,means=np.asarray(trainer.mu).tolist()))
if __name__=='__main__':main()
