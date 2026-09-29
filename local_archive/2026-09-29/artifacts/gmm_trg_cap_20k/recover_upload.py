"""Recover post-training W&B upload failures; never import or run the learner."""
import json
import os
from pathlib import Path
import time
import numpy as np
from flax.serialization import msgpack_restore
from dotenv import load_dotenv
import wandb

load_dotenv('/home/heechan/.env',override=False)
root=Path('/home/heechan/optiq-experiments/trg-cap-20k-20260921')
for path in sorted(root.glob('*-s0.json')):
    job=json.loads(path.read_text())
    if job['status']!='failed':continue
    log=(root/(path.stem+'.log')).read_text()
    assert 'run.log_artifact(artifact)' in log and 'wandb.errors' in log
    configs=list((root/'outputs').glob(path.stem+'_*/config.json'))
    assert len(configs)==1
    out=configs[0].parent
    cfg=json.loads(configs[0].read_text())
    checkpoints=list(out.rglob('*_20000.msgpack'))
    assert {p.name for p in checkpoints}=={'actor_state_20000.msgpack','critic_state_20000.msgpack'}
    for p in checkpoints:
        state=msgpack_restore(p.read_bytes())
        assert int(state['step'])==15000
    summary=dict(completed=True,timesteps=20000,updates=15000,last_eval_step=20000,
        logging_recovered=True,artifact_upload_fallback='run_files',
        recovery_note='Training and final evaluation completed; only post-training artifact upload failed. No learner updates or evaluations were rerun.')
    for mode in ('zero_z','stochastic_z'):
        files=list(out.rglob(f'evaluations_{mode}.npz'));assert len(files)==1
        with np.load(files[0]) as d:
            assert d['timesteps'].tolist()==[1]+list(range(1000,20001,1000))
            assert d['results'].shape==(21,10) and np.isfinite(d['results']).all()
            summary[f'final_eval_return_{mode}']=float(d['results'][-1].mean())
    summary['final_eval_return']=summary['final_eval_return_zero_z']
    dirs=list((out/'wandb').glob('run-*'));assert len(dirs)==1
    run_id=dirs[0].name.rsplit('-',1)[1]
    run=wandb.init(entity='OptiQ',project='abla',id=run_id,resume='must',
        dir=str(root),settings=wandb.Settings(init_timeout=120))
    run.summary.update(summary)
    for file in [configs[0],*checkpoints,*out.rglob('evaluations_*.npz')]:
        run.save(str(file),base_path=str(out),policy='now')
    url=run.url
    run.finish(exit_code=0)
    result=dict(wandb_url=url,timesteps=20000,updates=15000,logging_recovered=True)
    (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    job.update(result,run_dir=str(out),original_exit_code=job['exit_code'],
        status='completed',logging_recovered_at=time.time())
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(job,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(dict(task=job['task'],sigma=job['sigma'],recovered=True,url=url)),flush=True)
