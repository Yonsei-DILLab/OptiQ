from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1]/'gmm40_fixed_fresh'
out=Path(__file__).resolve().parent/'previous_mu'
a=np.load(out/'evaluations.npz');t=json.loads((root/'target.json').read_text())
means=np.asarray(t['means']);std=np.asarray(t['std']);theta=np.linspace(0,2*np.pi,100)
def metrics(x,ref):
 def classify(v):
  d=(((v[:,None,:]-means)/std[None,:,None])**2).sum(-1);labels=d.argmin(-1);near=d.min(-1)<=9
  return near,np.bincount(labels[near],minlength=40)
 near,c=classify(x);_,rc=classify(ref)
 return near,dict(near=float(near.mean()),coverage=int((c>=np.maximum(10,.1*rc*len(x)/len(ref))).sum()),mass_tv=float(.5*np.abs(c/c.sum()-rc/rc.sum()).sum()))
fig,axs=plt.subplots(1,2,figsize=(12.5,7.2));summary=[]
for seed,ax in enumerate(axs):
 x=a[f'centers_s{seed}'];near,m=metrics(x,a[f'reference_s{seed}']);_,full=metrics(a[f'full_saved_s{seed}'],a[f'reference_s{seed}']);_,cpu=metrics(a[f'full_cpu_s{seed}'],a[f'reference_s{seed}'])
 summary.append(dict(seed=seed,mu_only=m,full_policy_saved=full,full_policy_cpu_check=cpu))
 for sel,c in [(~near,'#e9884f'),(near,'#2381b4')]:ax.scatter(*x[sel].T,s=.7,alpha=.36,c=c,rasterized=True)
 for center,s in zip(means,std):ax.plot(center[0]+3*s*np.cos(theta),center[1]+3*s*np.sin(theta),ls='--',lw=.6,c='#838b94')
 ax.scatter(*means.T,marker='+',s=26,c='#17242e')
 ax.set(xlim=(-42,42),ylim=(-42,42),aspect='equal',xlabel='Action x₁',ylabel='Action x₂')
 ax.grid(alpha=.12)
 ax.set_title(f"Fresh z · seed {seed} · 100k\nμ-only near {m['near']:.2%} · coverage {m['coverage']}/40\nFull-policy near {full['near']:.2%}",fontsize=12)
fig.suptitle('Previous GMM40: fresh-latent μ outputs, Gaussian noise removed',fontsize=18,fontweight='bold',y=.98)
fig.text(.5,.918,'All 32,768 independently sampled z shown per seed · N=M=64 · batch=256 · T=1 · mean init=1',ha='center',fontsize=11)
fig.subplots_adjust(top=.79,bottom=.18,left=.065,right=.985,wspace=.18)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within any GT 3σ'),Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside all GT 3σ')],loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
fig.text(.5,.035,'Historical tanh policy: x = 40 tanh(μ(z)), no σ ε. CPU checkpoint reevaluation; these are centers, not full-policy samples.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'previous_fresh_mu_only.{ext}',dpi=170)
(out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
