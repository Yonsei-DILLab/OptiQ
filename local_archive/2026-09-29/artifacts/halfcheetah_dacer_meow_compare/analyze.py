"""Read-only W&B evaluation comparison; no smoothing or interpolation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parent
histories=json.loads((ROOT/'histories.json').read_text())
streams={k:{} for k in ('OptiQ stochastic_z','OptiQ zero_z','MEOW deterministic')}
for run in histories:
    keys=run['keys']; mapping=[('MEOW deterministic',keys[1])] if run['project']=='baseline' else [
        ('OptiQ stochastic_z',keys[1]),('OptiQ zero_z',keys[2])]
    for label,metric in mapping:
        rows={int(r[keys[0]]):float(r[metric]) for r in run['history'] if keys[0] in r and metric in r and r[metric] is not None}
        assert rows and all(np.isfinite(v) for v in rows.values())
        assert run['seed'] not in streams[label], 'Duplicate seed/run'
        streams[label][run['seed']]=rows

def aggregate(label,seeds,lo,hi):
    expected=list(range(lo+5000,hi+1,5000)); assert len(expected)==(hi-lo)//5000
    values=np.array([[streams[label][s][x] for x in expected] for s in seeds])
    means=values.mean(axis=1)
    return dict(mean=float(means.mean()),seed_std=float(means.std(ddof=1)),
        per_seed={str(s):float(v) for s,v in zip(seeds,means)},
        evaluations_per_seed=len(expected),episodes_per_evaluation=10,
        final_mean=float(values[:,-1].mean()),final_seed_std=float(values[:,-1].std(ddof=1)),
        linear_slope_per_100k=float(np.polyfit(np.array(expected)/100000,values.mean(axis=0),1)[0]),
        mean_within_seed_temporal_std=float(values.std(axis=1,ddof=1).mean()))

seeds=sorted(set.intersection(*[{s for s,v in group.items() if max(v)>=1000000} for group in streams.values()]))
all_seeds=sorted(set.intersection(*[set(group) for group in streams.values()]))
common_end=min(max(group[s]) for group in streams.values() for s in all_seeds)
result=dict(completed_matched_seeds=seeds,all_seeds=all_seeds,common_end=common_end,
    criterion='900000 < env_steps <= 1000000; average 20 evaluations of 10 episodes within each seed, then average seeds. SD is sample SD of seed means.',
    evaluation_note='OptiQ stochastic_z uses random latent and conditional mean action; zero_z fixes latent zero. MEOW uses its logged deterministic evaluation.',
    final_100k={},common_100k={},runs=[{k:r[k] for k in ('project','seed','id','url','state','fetched_at')} for r in histories])
for label in streams:
    last=aggregate(label,seeds,900000,1000000)
    prev=aggregate(label,seeds,800000,900000)
    first50=aggregate(label,seeds,900000,950000)
    last50=aggregate(label,seeds,950000,1000000)
    last.update(previous_100k_mean=prev['mean'],change_from_previous_100k=last['mean']-prev['mean'],
        first_50k_mean=first50['mean'],last_50k_mean=last50['mean'],last_50k_change=last50['mean']-first50['mean'])
    result['final_100k'][label]=last
    result['common_100k'][label]=aggregate(label,all_seeds,common_end-100000,common_end)
o=result['final_100k']['OptiQ stochastic_z']; m=result['final_100k']['MEOW deterministic']
result['comparison']=dict(meow_minus_optiq=m['mean']-o['mean'],optiq_relative_to_meow_pct=100*(o['mean']/m['mean']-1),
    optiq_wins=sum(o['per_seed'][str(s)]>m['per_seed'][str(s)] for s in seeds))
(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')

colors={'OptiQ stochastic_z':'#136fc1','OptiQ zero_z':'#729ac3','MEOW deterministic':'#e78023'}
fig,axes=plt.subplots(1,2,figsize=(13.5,5),gridspec_kw={'width_ratios':[1.5,1]})
xs=np.arange(700000,1000001,5000)
for label in streams:
    vals=np.array([[streams[label][s][int(x)] for x in xs] for s in seeds])
    avg=vals.mean(axis=0); sd=vals.std(axis=0,ddof=1)
    axes[0].plot(xs,avg,label=label,color=colors[label],lw=2,ls='--' if label=='OptiQ zero_z' else '-')
    if label!='OptiQ zero_z':axes[0].fill_between(xs,avg-sd,avg+sd,color=colors[label],alpha=.13)
axes[0].axvspan(900000,1000000,color='#6b7785',alpha=.09)
axes[0].axvline(900000,color='#6b7785',lw=1,ls=':')
axes[0].set(title='Late learning curve (mean ± seed SD)',xlabel='Environment steps',ylabel='Evaluation return')
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'))
axes[0].legend(loc='lower right',frameon=False,fontsize=9)
labels=list(streams)
for i,label in enumerate(labels):
    d=result['final_100k'][label]
    axes[1].bar(i,d['mean'],width=.65,color=colors[label],alpha=.8)
    axes[1].errorbar(i,d['mean'],yerr=d['seed_std'],fmt='none',ecolor='#333333',capsize=5)
    vals=list(d['per_seed'].values())
    axes[1].scatter(i+np.linspace(-.14,.14,len(vals)),vals,c='white',edgecolors='#343434',s=34,zorder=4)
    axes[1].text(i,max(max(vals),d['mean']+d['seed_std'])+250,f"{d['mean']:,.0f} ± {d['seed_std']:,.0f}",ha='center',fontsize=10,weight='bold')
axes[1].set_xticks(range(3),['OptiQ\nstochastic_z','OptiQ\nzero_z','MEOW\ndeterministic'])
axes[1].set(title='Last 100k: 900k < step ≤ 1M',ylabel='Mean evaluation return')
axes[1].set_ylim(0,max(d['mean']+d['seed_std'] for d in result['final_100k'].values())+1300)
for ax in axes:
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('HalfCheetah-v4 · OptiQ GMM-TRG (T=0.25, β=1, DACER on) vs MEOW',fontsize=15,weight='bold')
fig.text(.5,.018,'Matched completed seeds '+', '.join(map(str,seeds))+'. Each seed: 20 evaluations × 10 episodes. OptiQ seed 4 is still running.\nBands/error bars: sample SD across seeds; dots: seed means. Evaluation policies differ as labeled.',ha='center',fontsize=9,color='#45515f')
fig.tight_layout(rect=[0,.09,1,.94])
fig.savefig(ROOT/'comparison.png',dpi=180)
fig.savefig(ROOT/'comparison.pdf')
print(json.dumps(result,indent=2))
