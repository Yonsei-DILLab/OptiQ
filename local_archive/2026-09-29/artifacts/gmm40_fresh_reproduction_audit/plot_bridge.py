from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1]
out=Path(__file__).resolve().parent
old=root/'gmm40_fixed_fresh'
cur=root/'gmm40_capm3_queue/interim/20260921-193206'
t=json.loads((cur/'target.json').read_text());means=np.asarray(t['means']);std=np.asarray(t['std'])
p=cur/'samples/capm3_mean1_optiq_trg_s0_100k/100000'
previous=np.load(old/'results/fresh_seed0/step100000.npz')['samples']
items=[('Previous tanh Gaussian\nfresh z · seed 0 · 100k',previous),('Current TRG · centers only (40μ)\nfresh z · seed 0 · 100k',np.load(p/'samples_mu_only.npy')),('Current TRG · full policy\nfresh z · seed 0 · 100k',np.load(p/'samples.npy'))]
fig,axs=plt.subplots(1,3,figsize=(16.5,6.5))
theta=np.linspace(0,2*np.pi,100)
stats=[]
for ax,(label,x) in zip(axs,items):
 near=((((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9)
 for sel,col in [(~near[:10000],'#e9884f'),(near[:10000],'#2381b4')]:ax.scatter(*x[:10000][sel].T,s=1,alpha=.5,c=col,rasterized=True)
 for center,s in zip(means,std):ax.plot(center[0]+3*s*np.cos(theta),center[1]+3*s*np.sin(theta),ls='--',lw=.5,c='#838b94')
 ax.scatter(*means.T,marker='+',s=22,c='#17242e')
 ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂',xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
 ax.set_title(f'{label}\nWithin GT 3σ: {near.mean():.2%}',fontsize=12)
 ax.grid(alpha=.12);stats.append(dict(label=label,near=float(near.mean()),samples=len(x)))
fig.suptitle('GMM40: mean-network bridges and conditional Gaussian spread',fontsize=18,fontweight='bold',y=.99)
fig.text(.5,.90,'Previous: log σ=[−5,+1], init σ=0.5, teacher floor=0.05   |   Current: log σ=[−5,−3], init=−3, floor=exp(−5)',ha='center',fontsize=10)
fig.subplots_adjust(top=.79,bottom=.20,wspace=.20,left=.045,right=.99)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within any GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside all GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.08),ncol=2,frameon=False)
fig.text(.5,.035,'10,000 unfiltered samples shown per panel. Previous rate uses 32,768 samples; current rates use 10,000. Same GT and mean init scale 1.',ha='center',fontsize=10)
fig.savefig(out/'fresh_mean_noise_comparison.png',dpi=170)
fig.savefig(out/'fresh_mean_noise_comparison.pdf')
(out/'visual_metrics.json').write_text(json.dumps(stats,indent=2))
print(out/'fresh_mean_noise_comparison.png')
