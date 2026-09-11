"""Generate partial or final five-seed comparisons without changing experiments."""
import argparse,csv,html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    root=Path(args.campaign);out=root/'comparison';out.mkdir(exist_ok=True)
    tasks=json.loads((root/'tasks.json').read_text());rows=[];histories={};failures=[];initial={}
    for task in tasks:
        folder=root/'runs'/task['name'];cfg=task['config']
        if (folder/'failed.json').exists(): failures.append(dict(name=task['name'],**json.loads((folder/'failed.json').read_text())))
        if (folder/'manifest.json').exists():
            initial[(cfg['case'],cfg['seed'],cfg['version'])]=json.loads((folder/'manifest.json').read_text())['initial_state_sha256']
        if (folder/'evaluations.jsonl').exists():
            history=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines() if line]
            histories[task['name']]=history
        if (folder/'COMPLETE').exists():
            completed=json.loads((folder/'completed.json').read_text())
            assert completed['finished'] and completed['updates']==cfg['updates']
            row=dict(completed['final_metrics'],training_seconds=completed['training_seconds'])
            row.pop('mean_per_coordinate');row.pop('std_per_coordinate');rows.append(row)
    pair_count=0
    for case,seed,version in initial:
        if version=='ver1' and (case,seed,'ver2') in initial:
            assert initial[(case,seed,'ver1')]==initial[(case,seed,'ver2')],(case,seed)
            pair_count+=1
    if rows:
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'per_seed.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    metrics=['mean_q','mean_abs_error','backup_k50_rmse','mode_bin_tv','effective_modes','sliced_w2','training_seconds']
    groups=[]
    for case in dict.fromkeys(t['config']['case'] for t in tasks):
        for version in ('ver1','ver2'):
            selected=[r for r in rows if r['case']==case and r['version']==version]
            if selected:
                group=dict(case=case,version=version,seeds=len(selected))
                for metric in metrics:
                    values=np.array([r[metric] for r in selected]);group[metric]=float(values.mean())
                    group[metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else None
                groups.append(group)
    status=dict(completed=len(rows),total=len(tasks),failures=failures,verified_initial_pairs=pair_count,
                groups=groups,partial=len(rows)!=len(tasks))
    (out/'summary.json').write_text(json.dumps(status,indent=2))
    sections=[]
    for case in dict.fromkeys(t['config']['case'] for t in tasks):
        fig,axes=plt.subplots(1,3,figsize=(13,3.3));seen=False
        for version,color in [('ver1','#b55a30'),('ver2','#2768b1')]:
            hs=[history for name,history in histories.items() if history and history[0]['case']==case and history[0]['version']==version]
            if not hs:continue
            seen=True
            for ax,metric in zip(axes,['mean_q','mode_bin_tv','effective_modes']):
                steps=sorted(set(row['update'] for history in hs for row in history))
                mean=[];std=[]
                for step in steps:
                    values=[row[metric] for history in hs for row in history if row['update']==step]
                    mean.append(np.mean(values));std.append(np.std(values))
                mean=np.array(mean);std=np.array(std)
                ax.plot(steps,mean,label=version,color=color)
                ax.fill_between(steps,mean-std,mean+std,color=color,alpha=.15)
                ax.set_xscale('symlog',linthresh=10);ax.set_xlabel('Actor updates');ax.set_title(metric)
            axes[0].axhline(hs[0][0]['reference_mean_q'],color='black',ls=':',lw=1)
        if seen:
            axes[0].legend();fig.suptitle(case);fig.tight_layout()
            fig.savefig(out/(case+'.png'),dpi=150);fig.savefig(out/(case+'.pdf'))
            sections.append('<h2>'+html.escape(case)+'</h2><img src="'+case+'.png">')
        plt.close(fig)
    table='<table><tr><th>case</th><th>version</th><th>seeds</th>'+''.join('<th>'+m+'</th>' for m in metrics)+'</tr>'
    for group in groups:
        table+='<tr><td>'+group['case']+'</td><td>'+group['version']+'</td><td>'+str(group['seeds'])+'</td>'+''.join('<td>'+format(group[m],'.5g')+'</td>' for m in metrics)+'</tr>'
    table+='</table>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Batch-one OT comparison</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style><h1>Batch-one OT: ver1 vs ver2</h1><p>'+str(len(rows))+' / '+str(len(tasks))+' completed. Temperature=1; unchanged frozen Q, bounded truncated KDE. GMM40 uses unbounded KDE. Equal updates, unequal sample/compute budgets. Five training seeds; shading describes currently available seed variation. GMM reference mean includes Monte Carlo uncertainty.</p>'+table+''.join(sections))
    print(json.dumps(dict(completed=len(rows),total=len(tasks),failures=failures,verified_initial_pairs=pair_count)),flush=True)
    if args.require_complete:
        assert len(rows)==len(tasks) and not failures and pair_count==70,'Incomplete or failed campaign'
        (root/'COMPLETE').write_text('140 runs; 70 paired initial states; finite final results verified\n')


if __name__=='__main__':main()
