"""Continuously export all seeds and mean/sample-SD curves; no best-run selection."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from .run import atomic


def update(plan_path, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads(plan_path.read_text());groups=defaultdict(list);raw=[]
    for j in plan['jobs']:
        if j['task'] not in ('hammer','pen'):continue
        path=Path(j['output'])/'evaluations.jsonl';records=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:e=json.loads(line)
                except ValueError:continue
                v=e[e['primary_mode']]
                row=dict(method=j['method'],task=j['task'],seed=j['seed'],steps=e['steps'],
                    success_rate=v['success_rate'],mean_return=v['mean_return'],
                    final_success_rate=v['final_success_rate'],success_25_steps_rate=v['success_25_steps_rate'])
                records.append(row);raw.append(row)
        groups[(j['task'],j['method'])].append((j,records))
    destination.mkdir(parents=True,exist_ok=True)
    if raw:
        with (destination/'all-seed-learning-curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    summaries=[];curves=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for (task,method),members in groups.items():
        assert len(members)==5 and {j['seed'] for j,_ in members}==set(range(5))
        maps=[{r['steps']:r for r in rows} for _,rows in members]
        common=sorted(set.intersection(*(set(m) for m in maps)))
        for step in common:
            values=np.array([m[step]['success_rate'] for m in maps])
            curves.append(dict(task=task,method=method,steps=step,n=5,mean=float(values.mean()),seed_sd=float(values.std(ddof=1))))
        rows=[r for r in curves if r['task']==task and r['method']==method]
        ax=axes[('hammer','pen').index(task)]
        if rows:
            x=np.array([r['steps']/1e6 for r in rows]);y=np.array([r['mean'] for r in rows]);sd=np.array([r['seed_sd'] for r in rows])
            ax.plot(x,y,label=method);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),alpha=.15)
        complete=all(j['status']=='completed' and (Path(j['output'])/'completed.json').exists() for j,_ in members)
        result=dict(task=task,method=method,seeds=list(range(5)),completed=sum(j['status']=='completed' for j,_ in members),
                    final_five_seed_report_ready=complete)
        if complete:
            assert all(5000192 in m for m in maps)
            for key in ('success_rate','mean_return','final_success_rate','success_25_steps_rate'):
                values=np.array([m[5000192][key] for m in maps]);result[key]=dict(mean=float(values.mean()),seed_sd=float(values.std(ddof=1)),n=5)
        summaries.append(result)
    for task,ax in zip(('Hammer','Pen'),axes):
        ax.set(title=task,xlabel='Online environment transitions (million)',ylim=(0,1));ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    axes[0].set_ylabel('Evaluation success rate (any current-env success)')
    fig.suptitle('Verified 25-demo dataset: five-seed mean ± sample SD; common evaluation steps only')
    fig.tight_layout();fig.savefig(destination/'learning-curves.png',dpi=180);fig.savefig(destination/'learning-curves.pdf');plt.close(fig)
    atomic(destination/'five-seed-summary.json',dict(observed_unix=time.time(),groups=summaries,
        all_20_completed=all(s['final_five_seed_report_ready'] for s in summaries),
        final_evaluation_episodes_per_seed=100,sd_definition='Sample standard deviation across the five training seeds (ddof=1)',
        curve_rule='All five seeds at common recorded steps; no selection, interpolation, or comparison of different step budgets'))
    atomic(destination/'five-seed-curves.json',curves)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    while True:
        update(a.plan,a.output)
        if json.loads((a.output/'five-seed-summary.json').read_text())['all_20_completed']:return
        time.sleep(60)


if __name__=='__main__':main()
