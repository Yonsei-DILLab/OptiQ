"""Generate quality/latency figures without selecting winners or hiding failures."""
import argparse,base64,json
from pathlib import Path
import numpy as np
from .evaluate import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=a.output/'report';folder.mkdir(exist_ok=True)
    queue=json.loads((a.output/'queue.json').read_text())
    rows=[]
    for j in queue:
        run=a.output/'runs'/f"{j['condition']}_s{j['seed']}"
        if (run/'latest.json').exists():rows.append(dict(j,**json.loads((run/'latest.json').read_text())))
    finished=[r for r in rows if r['status']=='completed' and 'latency' in r]
    atomic_json(folder/'summary.json',dict(queue=queue,results=rows))
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for method in dict.fromkeys(r['condition'] for r in finished):
        rs=[r for r in finished if r['condition']==method]
        x=np.array([r['latency']['single_action_us'] for r in rs])
        for ax,metric in zip(axs,['coverage','sliced_w2']):
            y=np.array([r[metric] for r in rs]);line=ax.errorbar(np.median(x),y.mean(),yerr=y.std(ddof=1) if len(y)>1 else 0,fmt='o',capsize=3,label=method)
            ax.scatter(x,y,s=15,alpha=.35,color=line[0].get_color())
    for ax,metric,higher in zip(axs,['coverage','sliced_w2'],[True,False]):
        means=[]
        for method in dict.fromkeys(r['condition'] for r in finished):
            rs=[r for r in finished if r['condition']==method]
            means.append((float(np.median([r['latency']['single_action_us'] for r in rs])),float(np.mean([r[metric] for r in rs]))))
        frontier=[];best=-float('inf')
        for x,y in sorted(means):
            utility=y if higher else -y
            if utility>best:frontier.append((x,y));best=utility
        if len(frontier)>1:ax.plot(*np.asarray(frontier).T,ls='--',color='gray',alpha=.6,label='Empirical nondominated means')
    for ax in axs:ax.set_xscale('log');ax.set_xlabel('Single-action inference latency (µs, log)');ax.grid(alpha=.25)
    axs[0].set(ylabel='Recovered components / 40 ↑',ylim=(-.02,1.05),title='A-1. Mode coverage–latency')
    axs[1].set(ylabel='Sliced W2 ↓ (physical action units)',title='A-2. Distribution error–latency')
    if finished:axs[1].legend(fontsize=8,loc='best')
    fig.savefig(folder/'quality_latency.png',dpi=160);fig.savefig(folder/'quality_latency.pdf');plt.close(fig)
    # Training curves: avoid conflating updates, wall-clock and energy calls.
    fig,axs=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    colors={m:f'C{i%10}' for i,m in enumerate(dict.fromkeys(r['condition'] for r in rows))}
    for r in rows:
        run=a.output/'runs'/f"{r['condition']}_s{r['seed']}"
        history=[json.loads(q.read_text()) for q in sorted(run.glob('evaluation_*.json'))]
        for ax,clock in zip(axs,['updates','train_seconds','Q_evaluations']):
            ax.plot([h[clock] for h in history],[h['sliced_w2'] for h in history],color=colors[r['condition']],alpha=.5,label=r['condition'] if r['seed']==0 else None)
            ax.set(xlabel=clock,ylabel='Sliced W2 ↓');ax.grid(alpha=.2)
    if rows:axs[0].legend(fontsize=7)
    fig.savefig(folder/'learning_curves.png',dpi=150);plt.close(fig)
    counts={s:sum(j['status']==s for j in queue) for s in set(j['status'] for j in queue)}
    text=['# GMM40 oracle-energy amortization', '',f'Run status: {counts}. Figures use completed runs only; incomplete runs remain in summary.json.','',
          '## Setting and reading guide','',
          r'One-state terminal bandit: $Q(a)=\log p^\star(a)$, $\alpha=1$. Learners see only energy queries; target samples/components are used exclusively in evaluation.', '',
          r'Coverage uses $\hat w_k=L^{-1}\sum_l r_k^\star(a_l)$ and counts $\hat w_k\ge0.25/40$. No nearest-center cutoff is applied. A broad distribution can score well on this metric, so inspect SWD and the sample plots too.', '',
          r'SWD is $[128^{-1}\sum_v L^{-1}\sum_l(\mathrm{sort}(a\cdot v)_l-\mathrm{sort}(a^\star\cdot v)_l)^2]^{1/2}$, using 32,768 native stochastic actions and fixed directions. No KDE and no mu-only evaluation.', '',
          'Inference measures a warmed batch-1 call, including randomness and synchronized host action transfer. JAX is JIT and PyTorch is eager. Training counts and architectures differ; read the protocol and all three training clocks before attributing differences to an objective.', '',
          'Bounded policies use [-50,50]². Their exact optimum is the box-conditioned target; the omitted full-GMM mass is recorded in provenance.json. DIPO is a Q-improvement diffusion comparator, not an exact-MaxEnt objective. MEow is a flow Q/V adapter, not SAC-NF.', '',
          '## Main figures','', '![Quality–latency](quality_latency.png)','', '![Training curves](learning_curves.png)','',
          '| Method | Seed | Status | Updates | Components /40 | SWD | Latency µs |', '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['condition']} | {r['seed']} | {r['status']} | {r['updates']} | {r['recovered_components']} | {r['sliced_w2']:.3f} | {r.get('latency',{}).get('single_action_us',float('nan')):.1f} |")
    text+=['','## Native action histograms','']
    for r in rows:
        path=a.output/'runs'/f"{r['condition']}_s{r['seed']}"/f"samples_{r['updates']:07d}.png"
        if path.exists():
            import shutil
            name=f"{r['condition']}_s{r['seed']}.png";shutil.copy2(path,folder/name);text+=['![Sample histogram]('+name+')','']
    md='\n'.join(text)+'\n';(folder/'report.md').write_text(md)
    import markdown
    body=markdown.markdown(md,extensions=['tables'])
    for image in folder.glob('*.png'):
        body=body.replace('src="'+image.name+'"','src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"')
    (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>GMM40 oracle-energy comparison</title><style>body{max-width:1300px;margin:40px auto;font:17px system-ui;line-height:1.6}img{max-width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'+body)
if __name__=='__main__':main()
