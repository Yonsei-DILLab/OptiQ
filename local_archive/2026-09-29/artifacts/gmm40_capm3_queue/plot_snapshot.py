"""Show all saved policy samples, with each panel's actual evaluation step."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('snapshot',type=Path)
p.add_argument('--baseline',type=Path,required=True)
args=p.parse_args()
new=json.loads((args.snapshot/'snapshot.json').read_text())
old=json.loads((args.baseline/'snapshot.json').read_text())
target=json.loads((args.snapshot/'target.json').read_text())
oldtarget=json.loads((args.baseline/'target.json').read_text())
for key in ('means','std','weights'):
    np.testing.assert_array_equal(target[key],oldtarget[key])
means=np.array(target['means']);std=np.array(target['std'])
out=args.snapshot/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False})
blue,orange,ink='#2381b4','#e9884f','#17242e'
theta=np.linspace(0,2*np.pi,100)


def panel(ax,root,run,label):
    metric=run['latest'];step=metric['step']
    samples=np.load(root/'samples'/run['job']['name']/str(step)/'samples.npy')
    assert samples.shape==(10000,2) and np.isfinite(samples).all()
    distance=(((samples[:,None,:]-means)/std[None,:,None])**2).sum(-1)
    near=distance.min(-1)<=9
    assert np.isclose(near.mean(),metric['high_density_fraction'])
    for selected,color in ((~near,orange),(near,blue)):
        ax.scatter(*samples[selected].T,s=1.4,c=color,alpha=.4,rasterized=True)
    for center,sigma in zip(means,std):
        ax.plot(center[0]+3*sigma*np.cos(theta),center[1]+3*sigma*np.sin(theta),
                color='#818990',ls='--',lw=.65)
    ax.scatter(*means.T,marker='+',s=28,c=ink,lw=1.15)
    ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂',
           xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
    ax.grid(alpha=.12)
    state='final' if step==100000 else 'interim'
    ax.set_title(f"{label} · seed {run['job']['seed']} · {step//1000}k ({state})\n"
                 f"coverage {metric['mode_coverage']}/40 · near {near.mean():.2%}\n"
                 f"MMD² {metric['mmd2']:.4f} · mass TV {metric['mode_mass_tv']:.3f}",fontsize=12,pad=12)


def figure(items,title,subtitle,name,footnote):
    cols=min(2,len(items));rows=(len(items)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(6.2*cols,6.2*rows+1.5),squeeze=False)
    for ax,item in zip(axes.flat,items):panel(ax,*item)
    for ax in list(axes.flat)[len(items):]:ax.axis('off')
    fig.suptitle(title,fontsize=19,fontweight='bold',y=.982)
    fig.text(.5,.936,subtitle,ha='center',fontsize=11,color='#495866')
    fig.subplots_adjust(top=.82,bottom=.18,left=.065,right=.98,wspace=.22,hspace=.46)
    handles=[Line2D([],[],marker='o',linestyle='',color=blue,label='Within any GT 3σ'),
             Line2D([],[],marker='o',linestyle='',color=orange,label='Outside all GT 3σ'),
             Line2D([],[],marker='+',linestyle='',color=ink,label='GT component centers')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.088),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.063,'All 10,000 IID full-policy samples per panel · dashed circles: GT 3σ',ha='center',fontsize=9)
    fig.text(.5,.032,footnote,ha='center',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=160,facecolor='white')
    plt.close(fig)


nruns=sorted(new['runs'],key=lambda r:r['job']['seed'])
for run in nruns:
    c=run['config']
    assert c['actor_log_std_bounds']==[-5.,-3.] and c['initial_log_std']==-3.
    assert c['mean_output_init_scale']==1. and c['n']==c['m']==64 and c['batch']==256
figure([(args.snapshot,r,'Cap −3 / mean init 1') for r in nruns],
       'GMM40 · OptiQ Direct GMM / TRG · latest saved samples',
       'log σ=[−5, −3] · initial log σ=−3 · Xavier mean init=1 · N=M=64 · T=1 · batch=256',
       'capm3_latest_seeds',
       'Each panel uses its labeled update count. Unfinished seeds are not final four-seed results.')
old0=next(r for r in old['runs'] if r['job']['method']=='optiq_trg' and r['job']['seed']==0)
new0=next(r for r in nruns if r['job']['seed']==0)
assert old0['latest']['step']==new0['latest']['step']==100000
figure([(args.baseline,old0,'Original'),(args.snapshot,new0,'New setting')],
       'GMM40 · seed 0 · 100k updates · original vs new setting',
       'Original: log σ=[−5, −1], init −1, mean init 1e−4   |   New: log σ=[−5, −3], init −3, mean init 1',
       'seed0_before_after_100k',
       'Both sigma settings and mean initialization changed; their individual effects are not isolated.')
keys=('mode_coverage','high_density_fraction','mmd2','mode_mass_tv')
summary=dict(source_commit=new['manifest']['source_commit'],captured_at=new['captured_at'],
             latest=[dict(seed=r['job']['seed'],step=r['latest']['step'],**{k:r['latest'][k] for k in keys}) for r in nruns],
             seed0_baseline={k:old0['latest'][k] for k in keys})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(out)
