import argparse,json,os,pickle,signal,time,hashlib
from pathlib import Path
import numpy as np
import jax
from .core import Trainer,load_config,configurations
from .evaluate import evaluate
STOP=False

def stop(*_):
    global STOP
    STOP=True


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    cfg=configurations(load_config())[a.index];out=a.root/'runs'/cfg['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():print('Already complete',cfg['name'],flush=True);return
    manifest=json.loads((a.root/'SOURCE_MANIFEST.json').read_text());cfg['source_commit']=manifest['commit']
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for s in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
    exp=Trainer(cfg);step=0;start=time.time();ckpt=out/'checkpoint.pkl'
    if ckpt.exists():
        saved=pickle.loads(ckpt.read_bytes());exp.state=jax.tree_util.tree_map(jax.numpy.asarray,saved['state']);step=saved['step']
        assert saved['source_commit']==manifest['commit']
    else:np.savez(out/'initial_parameters.npz',params=np.asarray(exp.state[0]))
    with (out/'events.jsonl').open('a') as events:
        events.write(json.dumps(dict(event='start_or_resume',step=step,time=time.time(),job=os.environ.get('SLURM_JOB_ID'),source_commit=manifest['commit'],devices=str(jax.devices())))+'\n');events.flush()
        while True:
            if step in cfg['eval_steps']:
                for i,seed in enumerate(cfg['seeds']):
                    sd=out/f'seed{seed}';sd.mkdir(exist_ok=True)
                    if not (sd/f'metrics_{step:06d}.json').exists():evaluate(np.asarray(exp.state[0][i]),cfg,sd,step,seed)
            if step>=cfg['steps'] or STOP:break
            n=min(cfg['train_block'],cfg['steps']-step);t=time.time();exp.state,vals=exp.advance(exp.state,n);jax.block_until_ready(exp.state);step+=n
            assert np.isfinite(np.asarray(vals)).all() and np.isfinite(np.asarray(exp.state[0])).all(),'Nonfinite training'
            row=dict(step=step,nll=np.asarray(vals[-1]).tolist(),seconds_per_update=(time.time()-t)/n,time=time.time())
            atomic(out/'status.json',(json.dumps(row)+'\n').encode())
            if step%cfg['checkpoint_interval']==0 or STOP:
                atomic(ckpt,pickle.dumps(dict(state=jax.device_get(exp.state),step=step,source_commit=manifest['commit'])))
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(cfg['name'],row,flush=True)
                np.savez_compressed(out/f'parameters_{step:06d}.npz',params=np.asarray(exp.state[0]))
        atomic(ckpt,pickle.dumps(dict(state=jax.device_get(exp.state),step=step,source_commit=manifest['commit'])))
        if step==cfg['steps']:
            result=dict(step=step,time=time.time(),elapsed_seconds=time.time()-start,source_commit=manifest['commit'],seeds=cfg['seeds'])
            atomic(out/'COMPLETE.json',(json.dumps(result,indent=2)+'\n').encode())
            print('COMPLETE',cfg['name'],flush=True)
        else:events.write(json.dumps(dict(event='interrupted',step=step,time=time.time()))+'\n')
if __name__=='__main__':main()
