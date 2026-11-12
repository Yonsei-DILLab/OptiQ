from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;d=json.loads((P/'adroit-checkpoint-q.json').read_text())
fig,axes=plt.subplots(1,4,figsize=(15.5,4.6),sharey=True)
colors={(64,64):'#277DA8',(64,256):'#D9583F',(16,64):'#319269'}
for ax,task in zip(axes,['door','hammer','pen','relocate']):
 for nm,color in colors.items():
  for seed in [5,6]:
   rr=sorted([r for r in d['rows'] if (r['task'],r['N'],r['M'],r['seed'])==(task,*nm,seed)],key=lambda r:r['step'])
   ax.plot([r['step']/1e6 for r in rr],[r['q_min_mean'] for r in rr],marker='o',ms=4,ls='-' if seed==5 else '--',color=color,lw=1.3,label=f'N{nm[0]}/M{nm[1]}, s{seed}')
 ax.axhline(1000,color='#555',ls=':',label='Return upper bound')
 ax.set_title(task.title());ax.set_xlabel('Checkpoint environment steps (M)');ax.grid(alpha=.18);ax.set_xlim(.75,5.2);ax.set_ylim(450,2700)
 ax.legend(fontsize=6.3,loc='upper left',ncol=2)
axes[0].set_ylabel('Mean min-Q on identical expert transitions')
fig.suptitle('Adroit critic checkpoints | same 500 state-action pairs per task | seeds5/6\n20 evenly spaced transitions from each verified expert episode; frozen critics; no optimizer updates.',fontsize=12)
fig.tight_layout(rect=(0,0,1,.84));fig.savefig(P/'adroit-shared-state-q.png',dpi=180);fig.savefig(P/'adroit-shared-state-q.pdf')
print('saved')
