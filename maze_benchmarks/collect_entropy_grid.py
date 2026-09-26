"""Verify and compare entropy sweeps; never turn incomplete runs into results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .collect_deadline import digest, sync, snapshot
from .deadline_queue import verify
from .entropy_followup import TRAINING_SOURCE, SIMPLE_CAMPAIGN
from .run_nway_job import atomic_json
from .visualize_pointmaze import plot_map, plot_rollouts

HOST = 'vast-heechan-46'
BASE = '/home/heechan/optiq-experiments/'


def archive(root, campaign, expected):
    remote = BASE + campaign
    snap = snapshot(HOST, remote=remote)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'snapshot.json', snap)
    manifest = root / 'archive-manifest.json'
    saved = json.loads(manifest.read_text())['runs'] if manifest.exists() else {}
    if snap['queue'] is None:
        return saved
    if snap['queue']['source_commit'] != TRAINING_SOURCE:
        raise ValueError('source mismatch')
    for job in snap['queue']['jobs']:
        name = job['name']
        if name not in expected or any(job.get(k) != v for k,v in expected[name].items()):
            raise ValueError('unapproved job configuration')
        if job['state'] != 'complete' or name in saved:
            continue
        dest=root/'runs'/name
        dest.mkdir(parents=True, exist_ok=True)
        sync(f'{HOST}:{remote}/runs/{name}/',dest)
        proofpath=root/'proofs'/f'{name}.json'
        sync(f'{HOST}:{remote}/proofs/{name}-runs.json',proofpath)
        proof=json.loads(proofpath.read_text())
        for path,sha in proof['sha256'].items():
            if digest(dest/path)!=sha: raise ValueError(f'SHA mismatch {name}/{path}')
        if verify(dest,job,TRAINING_SOURCE,False)!=proof: raise ValueError('local proof mismatch')
        saved[name]=proof
        atomic_json(manifest,dict(runs=saved,complete=len(saved),expected=len(expected)))
    for filename in ('manifest.json','queue.json','registration.json'):
        sync(f'{HOST}:{remote}/{filename}',root/filename)
    atomic_json(manifest,dict(runs=saved,complete=len(saved),expected=len(expected)))
    return saved


def render(root, controls, plan, saved):
    fig,axes=plt.subplots(2,4,figsize=(16,9),constrained_layout=True)
    curves,cax=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
    rows=[]
    for row,method in enumerate(('mfpo','meow')):
        jobs=[dict(name=f'pm_simple-{method}-s0',control=True)] + [j for j in plan['jobs'] if j['method']==method]
        for col,job in enumerate(jobs):
            folder=(controls if col==0 else root)/'runs'/job['name']
            ax=axes[row,col];plot_map(ax,'simple')
            if col and job['name'] not in saved:
                ax.set_title(job['name']+'\npending',fontsize=9);continue
            config=json.loads((folder/'config.json').read_text())
            setting=(config['agent']['target_entropy_coeff'] if method=='mfpo' else config['agent']['alpha'])
            label=f"{'target H/d' if method=='mfpo' else 'alpha'}={setting:g}"
            records=[json.loads(p.read_text()) for p in sorted((folder/'evaluations').glob('*_summary.json'))]
            final=records[-1];data=final['policy']
            actual=plot_rollouts(ax,folder/'evaluations'/f"{final['step']:09d}_policy.npz")
            counts=actual['goals']+[0]*(4-len(actual['goals']))
            if counts!=data['goals']: raise ValueError('raw goal counts differ')
            ax.set_title(f'{method.upper()} {label}\nSuccess {data["success"]:.1%} · {data["reachable_goals"]}/4 goals')
            ax.set_xlabel(f'Goals {counts}\nFailures {data["failure"]}/{data["episodes"]}')
            for ri,key in enumerate(('success','reachable_goals')):
                cax[ri,row].plot([r['step']/1000 for r in records],[r['policy'][key] for r in records],label=label,lw=2.4)
            diagnostics=[r for r in records if 'entropy_diagnostics' in r]
            for ri,key in ((2,'alpha'),(3,'entropy_estimate')):
                if diagnostics:
                    cax[ri,row].plot([r['step']/1000 for r in diagnostics],[r['entropy_diagnostics'][key] for r in diagnostics],label=label,lw=2.4)
            rows.append(dict(run=job['name'],method=method,setting=setting,control=col==0,**data))
        for ri,ylabel in enumerate(('Success rate','Goals reached','Native alpha','Native entropy estimate')):
            cax[ri,row].set(xlabel='Transitions (k)',ylabel=ylabel,title=method.upper())
            cax[ri,row].grid(alpha=.2)
            if cax[ri,row].lines:cax[ri,row].legend()
    fig.suptitle('PointMaze Simple · direct policy samples · seed0 · final500 episodes\nExisting native controls vs entropy sensitivity; trajectories show first100')
    curves.suptitle('Same1M interaction budget; native entropy estimates differ between methods\nControl entropy history was not logged; MFPO entropy is before hard action clipping')
    out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'trajectories.png',dpi=170,bbox_inches='tight');plt.close(fig)
    curves.savefig(out/'curves.png',dpi=160,bbox_inches='tight');plt.close(curves)
    atomic_json(root/'results.json',dict(rows=rows,complete=len(saved),expected=6,single_seed=True,
        reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    a=p.parse_args();plan=json.loads(Path(__file__).with_name('POINTMAZE_ENTROPY_PLAN.json').read_text())
    saved=archive(a.output,SIMPLE_CAMPAIGN,{j['name']:j for j in plan['jobs']})
    render(a.output,a.controls,plan,saved)
    selection=a.output/'selection';selection.mkdir(exist_ok=True)
    sync(f'{HOST}:{BASE}pointmaze-entropy-selection-20260926/',selection)
    followups={}
    for method in ('mfpo','meow'):
        path=selection/f'plan-{method}.json'
        if path.exists():
            selected=json.loads(path.read_text())
            followups[method]=len(archive(a.output/'followups'/method,selected['campaign'],{j['name']:j for j in selected['jobs']}))
    print(json.dumps(dict(simple_complete=len(saved),simple_expected=6,followups=followups)))


if __name__=='__main__':main()
