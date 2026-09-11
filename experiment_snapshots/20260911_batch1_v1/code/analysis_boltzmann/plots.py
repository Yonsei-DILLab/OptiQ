"""Regenerate all figures from saved arrays; never runs training."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t

def save(fig,path):
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(str(path)+'.'+ext,dpi=170,bbox_inches='tight')
    plt.close(fig)

def run(root,out):
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for file in root.rglob('estimates.csv'):
        if not (file.parent/'COMPLETE').exists(): continue
        rows=list(csv.DictReader(file.open())); last=max(int(r['step']) for r in rows)
        for r in rows:
            if int(r['step']) not in (0,last): continue
            if int(r['step'])==0 and not r['method'].startswith('local_'): continue
            key=(r['case'],r['initialization'],r['method'],int(r['k']))
            grouped[key].append(r)
    cases=sorted(set((k[0],k[1]) for k in grouped))
    for case,init in cases:
        fig,ax=plt.subplots(1,2,figsize=(12,4))
        methods=sorted({k[2] for k in grouped if k[:2]==(case,init)})
        for method in methods:
            ks=sorted(k[3] for k in grouped if k[:3]==(case,init,method)); means=[]; rmses=[]; errors=[]
            for k in ks:
                rows=grouped[(case,init,method,k)]; v=np.array([float(r['mean']) for r in rows]); means.append(v.mean()); rmses.append(np.mean([float(r['rmse']) for r in rows]))
                errors.append(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0)
            ax[0].errorbar(ks,means,yerr=errors,marker='.',label=method); ax[1].plot(ks,rmses,marker='.',label=method)
        truth=float(rows[0]['truth']); ax[0].axhline(truth,color='black',ls='--',label='quadrature truth')
        for a in ax: a.set_xscale('log'); a.set_xlabel('Q evaluations / estimate'); a.grid(alpha=.2)
        ax[0].set_ylabel('Mean estimated backup'); ax[1].set_ylabel('RMSE'); ax[1].set_yscale('log'); handles,labels=ax[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(1.0,.96),fontsize=7)
        steps=max(int(r['step']) for key,rs in grouped.items() if key[:2]==(case,init) for r in rs)
        fig.suptitle(f'{case} / {init} / actor updates={steps} — seed-level 95% CI')
        save(fig,out/f'backup_{case}_{init}')
    for init in ('default','coverage'):
        selected=[k for k in grouped if k[0].startswith('asymmetric_') and k[1]==init and k[3]==50]
        if selected:
            methods=sorted({k[2] for k in selected}); fig,axs=plt.subplots(1,len(methods),figsize=(3*len(methods),3.5),squeeze=False)
            matrices=[]
            for method in methods:
                matrix=np.full((2,3),np.nan)
                for k in selected:
                    if k[2]!=method: continue
                    _,sep,width=k[0].split('_'); i=[.08,.16].index(float(width)); j=[.4,.8,1.3].index(float(sep))
                    matrix[i,j]=np.mean([float(r['bias']) for r in grouped[k]])
                matrices.append(matrix)
            vmax=max(float(np.nanmax(abs(m))) for m in matrices)+1e-12
            for ax,method,matrix in zip(axs[0],methods,matrices):
                im=ax.imshow(matrix,origin='lower',vmin=-vmax,vmax=vmax,cmap='RdBu_r',aspect='auto'); ax.set_xticks(range(3),['0.4','0.8','1.3']); ax.set_yticks(range(2),['0.08','0.16']); ax.set(xlabel='Mode separation',ylabel='Narrow-mode width',title=method); fig.colorbar(im,ax=ax)
            fig.suptitle('Signed backup bias, K=50 / '+init); save(fig,out/f'bias_heatmap_{init}')
    for file in root.rglob('distribution_*.npz'):
        if file.parent.name.endswith('seed0') or 'smoke' in str(file.parent):
            d=np.load(file); ref=np.load(file.parent/'reference.npz'); samples=d['samples']
            if samples.shape[1]<=2:
                fig,ax=plt.subplots(1,2,figsize=(9,3.5))
                if samples.shape[1]==1:
                    ax[0].hist(samples[:,0],bins=100,range=(-1,1),density=True,alpha=.5,label='actor')
                    x=ref['grid'][:,0]; ax[0].plot(x,ref['mass']/(x[1]-x[0]),label='target'); ax[0].legend()
                else: ax[0].hist2d(samples[:,0],samples[:,1],bins=80,range=[[-1,1],[-1,1]])
                x=np.arange(len(d['mode_mass'])); ax[1].bar(x-.2,d['reference_mode_mass'],.4,label='target'); ax[1].bar(x+.2,d['mode_mass'],.4,label='actor'); ax[1].legend(); ax[1].set_xlabel('Mode index')
                save(fig,out/(file.parent.name+'_'+file.stem))
    learning=defaultdict(list)
    for file in root.rglob('learning.json'):
        meta=json.loads((file.parent/'manifest.json').read_text()); learning[meta['arguments']['method']].append(json.loads(file.read_text()))
        probes=sorted(file.parent.glob('probe_*.npz'),key=lambda p:int(p.stem.split('_')[-1]))
        if meta['arguments']['seed']==0 and probes:
            mean=np.stack([np.load(p)['actions'].mean(1) for p in probes],1); left=np.stack([(np.load(p)['actions']<0).mean(1) for p in probes],1)
            fig,ax=plt.subplots(1,2,figsize=(10,4)); extent=[0,int(probes[-1].stem.split('_')[-1]),0,10]
            for a,im,title in zip(ax,(mean,left),('Mean action','P(action < 0)')):
                obj=a.imshow(im,origin='lower',aspect='auto',extent=extent); fig.colorbar(obj,ax=a); a.set_title(title); a.set_xlabel('Environment steps'); a.set_ylabel('Position')
            save(fig,out/f'policy_{meta["arguments"]["method"]}')
    if learning:
        fig,ax=plt.subplots(1,3,figsize=(14,4))
        for method,runs in learning.items():
            common=sorted(set.intersection(*[set(r['step'] for r in rows) for rows in runs]))
            for a,key in zip(ax,('return_mean','bias','good_occupancy')):
                v=np.array([[next(r[key] for r in rows if r['step']==s) for s in common] for rows in runs]); m=v.mean(0)
                ci=student_t.ppf(.975,len(v)-1)*v.std(0,ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.zeros_like(m)
                a.plot(common,m,label=method); a.fill_between(common,m-ci,m+ci,alpha=.15); a.set_ylabel(key); a.set_xlabel('Environment steps')
        ax[0].legend(); ax[1].axhline(0,color='gray',ls='--'); save(fig,out/'movecar_learning_bias')
    for file in root.rglob('landscape.npz'):
        d=np.load(file); fig,ax=plt.subplots(1,2,figsize=(10,4))
        for k in d.files:
            if k.endswith('_path'): ax[0].plot(d[k.replace('_path','_alpha')],d[k],label=k[:-5])
            if k.endswith('_r0.05'): ax[1].scatter(d[k][:,0],d[k][:,1],s=8,alpha=.5,label=k)
        ax[0].set(xlabel='Interpolation fraction',ylabel='Frozen-critic value objective'); ax[1].set(xlabel='L(theta + delta) - L(theta)',ylabel='L(theta - delta) - L(theta)')
        for a in ax: a.legend(fontsize=6); a.grid(alpha=.2); a.xaxis.set_major_locator(MaxNLocator(5))
        ax[1].ticklabel_format(axis='both',style='sci',scilimits=(-3,3))
        save(fig,out/f'landscape_{file.parent.name}')
    for file in root.rglob('landscape_20000.npz'):
        d=np.load(file); fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(d['alpha'],d['path']); ax[1].scatter(d['perturbation'][:,0],d['perturbation'][:,1],s=8,alpha=.5)
        ax[0].set(xlabel='Interpolation fraction',ylabel='Frozen-critic value objective'); ax[1].set(xlabel='Positive perturbation loss change',ylabel='Negative perturbation loss change'); save(fig,out/file.parent.name)
    for file in root.rglob('learned_backup.csv'):
        rows=list(csv.DictReader(file.open())); fig,ax=plt.subplots(figsize=(7,4))
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method and int(r['k'])==50]
            ax.plot([float(r['state']) for r in selected],[float(r['bias']) for r in selected],label=method)
        ax.axhline(0,color='black',ls='--'); ax.set(xlabel='MoveCar position',ylabel='Backup bias against common target Q'); ax.legend(); save(fig,out/file.parent.name)
    (out/'README.txt').write_text('Generated exclusively from saved arrays. CI across independent training seeds; no CI drawn for a single seed. Partial learning curves may be shown; frozen backup figures require COMPLETE.\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); run(a.root,a.out)
