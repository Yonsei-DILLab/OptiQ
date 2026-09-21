"""Backfill completed frozen runs affected by campaign_job's CLI abbreviation bug.

Only W&B metadata/history/artifacts and explicit local repair sidecars are changed.
Learner snapshots, checkpoints, evaluation files and existing logs are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .campaign import read, write, verified


def repair(root, manifest, job):
    import wandb
    folder=root/'results'/job['name']
    log=(root/'logs'/(job['name']+'.log')).read_text(errors='replace')
    ids=re.findall(r'https://wandb\.ai/OptiQ/gmm-trg/runs/([A-Za-z0-9]+)',log)
    if not ids:raise RuntimeError(f"No original W&B run ID for {job['name']}")
    run_id=ids[0]
    assert len(set(ids))==1,ids
    cfg=read(folder/'config.json')
    assert cfg['source_git_commit']==manifest['source_commit'] and cfg['name']==job['name']
    assert cfg['method']==job['method'] and cfg['seed']==job['seed']
    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    rows=sorted({r['step']:r for r in rows}.values(),key=lambda r:r['step'])
    assert rows[-1]['step']==job['steps']
    repair_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    wb=wandb.init(entity='OptiQ',project='gmm-trg',id=run_id,resume='must',
                  name=job['name'],group=manifest['plan']['name'],
                  job_type='gmm40-fixed-q-100k',tags=['gmm40','fixed-q',job['method'],'logging-repaired'],
                  dir=str(root/'wandb'),config=None)
    try:
        wb.name=job['name']
        wb.config.update({**cfg,'source_commit':cfg['source_git_commit'],
                           'logging_repair_source_commit':repair_commit},allow_val_change=True)
        wb.define_metric('updates')
        wb.define_metric('gmm40/*',step_metric='updates')
        wb.define_metric('train/*',step_metric='updates')
        for row in rows:
            wb.log({'updates':row['step'],
                    **{'gmm40/'+k:v for k,v in row.items() if isinstance(v,(int,float)) and k!='step'},
                    **{'train/'+k:v for k,v in row['training'].items() if isinstance(v,(int,float))}})
        final=folder/'evaluations'/f"step_{job['steps']:07d}"
        artifact=wandb.Artifact(job['name'],type='gmm40-result')
        for p in [folder/'config.json',folder/'latest.json',folder/'model_sizes.json',
                  folder/'update_count_audit.json',final/'samples.npy',final/'samples.png']:
            artifact.add_file(str(p),name=p.name)
        wb.log_artifact(artifact)
        wb.finish(exit_code=0)
    except BaseException:
        wb.finish(exit_code=1)
        raise
    api=wandb.Api()
    saved=api.run(f'OptiQ/gmm-trg/{run_id}')
    assert saved.name==job['name'] and saved.config['method']==job['method']
    assert saved.config['source_git_commit']==manifest['source_commit']
    assert saved.summary.get('updates')==job['steps'],dict(saved.summary)
    marker=dict(status='passed',run_id=run_id,url=f'https://wandb.ai/OptiQ/gmm-trg/runs/{run_id}',
                evaluations=len(rows),source_commit=manifest['source_commit'],
                logging_repair_source_commit=repair_commit,repaired_at=time.time())
    write(folder/'wandb_repair.json',marker)
    write(folder/'wandb_status.json',dict(url=marker['url'],warnings=[],logging_repair=marker))
    print(json.dumps(marker),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    import fcntl
    lock=(root/'wandb-repair.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(root/'manifest.json')
    # Scope this repair to the four newly approved cap=-3 runs.
    assert manifest['plan']['name']=='gmm40-trg-capm3-mean1-100k-4seed-20260921'
    assert len(manifest['jobs'])==4 and manifest['plan']['trg_actor']=={'log_std_max':-3.,'initial_log_std':-3.}
    try:
        while True:
            pending=[]
            for job in manifest['jobs']:
                marker=root/'results'/job['name']/'wandb_repair.json'
                if read(marker,{}).get('status')=='passed':continue
                state=read(root/'jobs'/(job['name']+'.json'),{})
                if state.get('status')=='failed':raise RuntimeError(f"Training failed: {job['name']}")
                if state.get('status')=='completed' and verified(root,job):repair(root,manifest,job)
                else:pending.append(job['name'])
            write(root/'wandb-repair-status.json',dict(status='waiting' if pending else 'completed',pending=pending,updated=time.time()))
            if not pending:return
            time.sleep(10)
    except BaseException as exc:
        write(root/'wandb-repair-failure.json',dict(error=repr(exc),updated=time.time()))
        raise


if __name__=='__main__':main()
