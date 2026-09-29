"""Compare completed seed-0 runs using their immutable 100k evaluations."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

base=Path(__file__).resolve().parent
root=base/'interim/20260921-181439'
snapshot=json.loads((root/'snapshot.json').read_text())
monitor=json.loads((base/'monitor-latest.json').read_text())
methods=['optiq_trg','mfpo','sac','sql']
eligible={j['method'] for j in monitor['jobs'] if j['seed']==0 and j['queue']['status']=='completed'}
assert eligible==set(methods), 'Refresh the snapshot and method list if more runs have finished.'
target=json.loads((root/'target.json').read_text())
means=np.array(target['means']);std=np.array(target['std']);theta=np.linspace(0,2*np.pi,100)
labels={'optiq_trg':'OptiQ Direct GMM','mfpo':'MFPO','sac':'SAC','sql':'SQL (JAX SVGD)'}
runs={r['job']['method']:r for r in snapshot['runs'] if r['job']['seed']==0}
out=root/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
blue='#2381b4';orange='#e9884f'
fig,axes=plt.subplots(2,2,figsize=(11,11.5))
rows=[]
for ax,m in zip(axes.flat,methods):
    run=runs[m];metric=run['latest']
    assert metric['step']==100000 and run['queue']['status']=='completed'
    samples=np.load(root/'samples'/run['job']['name']/'100000/samples.npy')
    assert samples.shape==(10000,2) and np.isfinite(samples).all()
    distances=np.sum(((samples[:,None,:]-means)/std[None,:,None])**2,axis=-1)
    near=distances.min(axis=1)<=9
    assert np.isclose(near.mean(),metric['high_density_fraction'])
    for selection,color in [(~near,orange),(near,blue)]:
        ax.scatter(*samples[selection].T,s=1.5,c=color,alpha=.4,rasterized=True)
    for center,sigma in zip(means,std):
        ax.plot(center[0]+3*sigma*np.cos(theta),center[1]+3*sigma*np.sin(theta),lw=.65,ls='--',color='#818990')
    ax.scatter(*means.T,marker='+',s=30,lw=1.1,c='#17242e')
    ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂',
           xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
    ax.grid(alpha=.12)
    ax.set_title(f"{labels[m]}\ncoverage {metric['mode_coverage']}/40 · near {near.mean():.1%}\n"
                 f"MMD² {metric['mmd2']:.4f} · mass TV {metric['mode_mass_tv']:.3f}",fontsize=12,pad=10)
    rows.append({'method':m,'seed':0,'actor_updates':100000,**{k:metric[k] for k in
        ['mode_coverage','high_density_fraction','mmd2','mode_mass_tv','sliced_wasserstein2']}})
fig.suptitle('GMM40 · seed 0 · completed 100k runs',fontsize=20,fontweight='bold',y=.984)
fig.text(.5,.95,'Fixed Q · T=1 · batch=256 · 10,000 full-policy samples per method',ha='center',fontsize=11)
fig.subplots_adjust(top=.875,bottom=.12,left=.075,right=.98,wspace=.24,hspace=.43)
handles=[Line2D([],[],marker='o',linestyle='',color=blue,label='Within any GT 3σ'),
         Line2D([],[],marker='o',linestyle='',color=orange,label='Outside all GT 3σ'),
         Line2D([],[],marker='+',linestyle='',color='#17242e',label='GT component centers')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.047),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.021,'One training seed; native model architectures and compute differ. Dashed circles: GT 3σ.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/f'completed_seed0_100k.{ext}',dpi=170,facecolor='white')
plt.close(fig)
with (out/'completed_seed0_100k.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(out/'completed_seed0_100k.json').write_text(json.dumps({'source_commit':snapshot['manifest']['source_commit'],
    'completion_checked_at':monitor['checked_at'],'rows':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
print(out/'completed_seed0_100k.png')
