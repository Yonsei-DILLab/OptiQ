import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from .core import mixture_np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.analysis.read_text());cfg=data['config'];rows=data['rows'];a.out.mkdir(parents=True,exist_ok=True)
    verified=[]
    for case in sorted(set(r['case'] for r in rows)):
        bad=[r for r in rows if r['case']==case and r['kind']=='bad_structure']
        if len(bad)==4 and all(r['strict_bad_minimum'] for r in bad):
            verified.append((min(r['diagnostic_hessian_eigenvalues'][0] for r in bad),case))
    if not verified:raise RuntimeError('No all-seed strict bad minimum; inspect diagnostics, do not label a saddle as a local minimum.')
    case=max(verified)[1];sub=[r for r in rows if r['case']==case];c=np.array(sub[0]['centers']);sigma=cfg['sigma']
    edges=np.linspace(c.min()-10*sigma,c.max()+10*sigma,cfg['histogram_bins']+1);xx=np.linspace(edges[0],edges[-1],10001)
    target,cdf=mixture_np(xx,c,sigma);targetmass=np.diff(mixture_np(edges,c,sigma)[1]);density={};evalrows=[]
    for kind in ['bad_structure','good_structure']:
        hist=[]
        for row in sub:
            if row['kind']!=kind:continue
            rng=np.random.default_rng(512000+row['seed']);mu=np.array(row['raw_means']);n=cfg['final_eval_samples']
            samples=mu[rng.integers(0,3,n)]+sigma*rng.normal(size=n)
            count=np.histogram(samples,bins=edges)[0];mass=count/n;hist.append(mass/np.diff(edges))
            den,modelcdf=mixture_np(xx,mu,sigma)
            evalrows.append(dict(kind=kind,seed=row['seed'],histogram_TV=float(.5*(np.abs(mass-targetmass).sum()+abs(1-mass.sum()))),population_W1=float(trapezoid(np.abs(modelcdf-cdf),xx)),sample_count=n))
        density[kind]=np.mean(hist,axis=0)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    fig,axes=plt.subplots(1,2,figsize=(12,3.9),sharex=True,sharey=True)
    for ax,kind,title in zip(axes,['bad_structure','good_structure'],['Bad-structure initialization','Good-structure initialization']):
        ax.plot(xx,target,'k--',lw=2,label='Target')
        ax.stairs(density[kind],edges,lw=1.5,color='#1f77b4',label='Learned GMM')
        ax.stairs(density[kind],edges,fill=True,alpha=.2,color='#1f77b4')
        means=np.array([r['raw_means'] for r in sub if r['kind']==kind]).mean(0)
        kl=np.mean([r['raw_kl'] for r in sub if r['kind']==kind])
        ax.set_title(title+'\n'+r'$\mathrm{KL}(p^*\Vert q)$'+f' = {kl:.4f}')
        ax.set_xlabel('x');ax.grid(alpha=.18);ax.legend(frameon=False);ax.set_xlim(c.min()-5*sigma,c.max()+5*sigma)
    axes[0].set_ylabel('Density')
    fig.suptitle('K = 3 | Equal fixed weights and variance | Mean-only gradient descent',fontsize=13)
    fig.tight_layout();fig.savefig(a.out/'mean_only_local_minimum.png');fig.savefig(a.out/'mean_only_local_minimum.pdf');plt.close(fig)
    hist=[json.loads(l) for l in (a.root/'runs'/case/'history.jsonl').read_text().splitlines()];step=np.array([r['step'] for r in hist]);m=np.array([r['means'] for r in hist]);v=np.array([r['nll'] for r in hist]);truth=sub[0]['raw_loss']-sub[0]['raw_kl']
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    for j in range(4):
        axes[0].plot(step,np.maximum(0,v[:,j]-truth),color='#d95f02',alpha=.75,label='Bad start' if j==0 else None)
        axes[0].plot(step,np.maximum(0,v[:,j+4]-truth),color='#1f77b4',alpha=.75,label='Good start' if j==0 else None)
    axes[0].set_ylabel(r'$\mathrm{KL}(p^*\Vert q)$');axes[0].set_xlabel('GD updates');axes[0].legend(frameon=False)
    for i in range(3):axes[1].plot(step,m[:,0,i],lw=1.5,label=rf'$\mu_{i+1}$')
    for val in c:axes[1].axhline(val,color='black',ls='--',lw=1,alpha=.5)
    axes[1].set_ylabel('Mean (bad-start seed 0)');axes[1].set_xlabel('GD updates');axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(a.out/'training_trajectory.png');fig.savefig(a.out/'training_trajectory.pdf');plt.close(fig)
    (a.out/'FIGURE_DATA.json').write_text(json.dumps(dict(case=case,selection='all four seeds strict bad minimum, largest minimum eigenvalue',evaluation=evalrows,selected_rows=sub),indent=2)+'\n')
    print(case);print(json.dumps(evalrows,indent=2))
if __name__=='__main__':main()
