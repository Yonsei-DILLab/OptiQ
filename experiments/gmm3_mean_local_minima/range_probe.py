import argparse,json,os,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .core import load_config,Trainer,adaptive


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('range_probe.json'));a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    pc=json.loads(a.config.read_text());cfg=load_config();cfg.update(steps=pc['steps'],learning_rate=pc['learning_rate'])
    case=dict(id=pc['study'],centers=pc['centers']);t=Trainer(case,cfg)
    rows=[dict(kind='explicit',index=i,initial_means=m) for i,m in enumerate(pc['fixed_initializations'])]
    rows += [dict(kind='uniform',seed=s,initial_means=np.random.default_rng(s).uniform(*pc['initialization_range'],3).tolist()) for s in pc['uniform_seeds']]
    initial=np.array([r['initial_means'] for r in rows]);assert ((initial>=pc['initialization_range'][0])&(initial<=pc['initialization_range'][1])).all();t.mu=jnp.array(initial)
    _,g=t.vg(t.mu)
    for i,r in enumerate(rows):r['initial_gradient']=np.asarray(g[i]).tolist()
    manifest=json.loads((a.root/'SOURCE_MANIFEST.json').read_text());commit=manifest['commit'];start=time.time()
    for step in range(0,cfg['steps']+1,cfg['block']):
        if step%cfg['checkpoint_interval']==0:
            v,g=t.vg(t.mu);status=dict(step=step,means=np.asarray(t.mu).tolist(),nll=np.asarray(v).tolist(),gradient_norm=np.linalg.norm(np.asarray(g),axis=-1).tolist())
            with (a.root/'history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            (a.root/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(step,flush=True)
        if step==cfg['steps']:break
        t.mu=t.advance(t.mu,cfg['block']);jax.block_until_ready(t.mu)
    truth=adaptive(case['centers'],case['centers'],cfg['sigma'])[0]
    for r,mu in zip(rows,np.asarray(t.mu)):
        val,grad,hess,err=adaptive(mu,case['centers'],cfg['sigma'])
        r.update(final_means=mu.tolist(),kl=val-truth,gradient_norm=float(np.linalg.norm(grad)),hessian_eigenvalues=np.linalg.eigvalsh(hess).tolist(),integration_error_estimate=err)
    result=dict(commit=commit,job=os.environ.get('SLURM_JOB_ID'),config=pc,base_config=cfg,rows=rows,elapsed=time.time()-start,steps=cfg['steps'])
    np.savez(a.root/'checkpoint.npz',initial_means=initial,means=np.asarray(t.mu),step=cfg['steps'],commit=commit)
    (a.root/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
