"""Scientific plots and offline Korean report; reads immutable diagnostics only."""
import argparse,base64,hashlib,json,re,subprocess,warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import markdown
from analyze import SIZES,STEPS,BRANCHES,collect,analyze,metrics,target_mass,write,clean

COLORS=['#087eaf','#e06c36','#1b9e77','#9564b6']
MODECOLORS=['#2076b6','#e59a35','#3c9b73','#a9afb8']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','axes.grid':False})

def title(n,m):return f'N={n}, M={m}'
def short(n,m):return f'{n}×{m}'
def subplots():return plt.subplots(4,2,figsize=(14,14),constrained_layout=True)
def finish(fig,path):
    temporal=any(k in path.name for k in ['learning_curves','role_emergence','trajectories'])
    stamp='0–20K updates' if temporal else '20K checkpoint'
    note='32768 fresh action samples/run; 256 bins; no smoothing' if 'histogram' in path.name else 'fixed evaluation latents=2048; no training modification'
    fig.text(.5,-.005,stamp+' | seeds 0–3 | '+note,ha='center',fontsize=8,color='#52606b')
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
def mean_finite(a):
    a=np.asarray(a,float);count=np.isfinite(a).sum(0)
    return np.divide(np.nansum(a,0),count,out=np.full(a.shape[1:],np.nan),where=count>0)
def matrix(ax,a,limit=None,cmap='RdBu_r',annotate=True,positive=False):
    a=np.asarray(a);lim=limit or max(np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0,1e-8)
    im=ax.imshow(np.ma.masked_invalid(a),cmap='Blues' if positive else cmap,vmin=0 if positive else -lim,vmax=lim)
    ax.set(xticks=range(a.shape[1]),xticklabels=['L','C','R'][:a.shape[1]],yticks=range(a.shape[0]))
    if annotate:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                x=a[i,j];s='N/A' if not np.isfinite(x) else (f'{x:.2f}' if lim<=1.01 else f'{x:.1f}')
                ax.text(j,i,s,ha='center',va='center',fontsize=10,color='white' if np.isfinite(x) and abs(x)>.55*lim else 'black')
    return im

def plot_all(runs,figdir):
    groups={(n,m):[r for r in runs if (r['n'],r['m'])==(n,m)] for n,m in SIZES}
    zlimit=max(float(np.abs(r['diags'][20000]['z']).max()) for r in runs)*1.02
    x=np.linspace(-1,1,2001);rho=sum(np.exp(-.5*((x-c)/.1)**2) for c in [-.6,0,.6])/(3*.1*np.sqrt(2*np.pi));rho=rho  # Normalize the target on [-1,1] below.
    from analyze import cdf
    rho/=float(cdf(1)-cdf(-1))
    fig,axs=subplots()
    for ax,(n,m) in zip(axs.flat,SIZES):
        ax.plot(x,rho,'--',color='#182936',lw=2,label='Exact target',zorder=10)
        for r in groups[n,m]:
            e=r['evals'][20000];ax.stairs(e['histogram']/np.diff(e['edges']),e['edges'],lw=1,alpha=.8,color=COLORS[r['seed']],label=f'seed {r["seed"]}')
        ax.set(title=title(n,m),xlim=(-1,1),ylim=(0,1.75),xlabel='Action',ylabel='Density');ax.legend(fontsize=8,ncol=3)
    finish(fig,figdir/'01_all_histograms.png')
    # Display all 32 individual outcomes without seed averaging.
    fig,axs=plt.subplots(8,4,figsize=(16,23),sharex=True,sharey=True,constrained_layout=True)
    for i,(n,m) in enumerate(SIZES):
        for r in groups[n,m]:
            ax=axs[i,r['seed']];e=r['evals'][20000];ax.plot(x,rho,'k--',lw=1.2);ax.stairs(e['histogram']/np.diff(e['edges']),e['edges'],lw=.7,color=COLORS[r['seed']]);ax.set(title=f'{short(n,m)} | seed {r["seed"]}',xlim=(-1,1),ylim=(0,1.75))
    finish(fig,figdir/'02_individual_histograms.png')
    fig,axs=subplots()
    for ax,(n,m) in zip(axs.flat,SIZES):
        for r in groups[n,m]:ax.plot([h['step'] for h in r['history']],[h['histogram_tv'] for h in r['history']],color=COLORS[r['seed']],label=f'seed {r["seed"]}')
        ax.axhline(.1,color='#777',ls=':',lw=.8);ax.set(title=title(n,m),xlabel='Training update',ylabel='Histogram TV',ylim=(0,.43));ax.legend(fontsize=8,ncol=2)
    finish(fig,figdir/'03_learning_curves.png')
    for key,name,ylims,ylabel in [('mu','04_latent_means',(-.85,.85),'Representative tanh(mu)'),('log_sigma','05_latent_sigmas',(0,.85),'Pre-tanh sigma')]:
        fig,axs=subplots()
        for ax,(n,m) in zip(axs.flat,SIZES):
            for r in groups[n,m]:
                d=r['diags'][20000];order=np.argsort(d['z'][:,0]);y=np.tanh(d[key][:,0]) if key=='mu' else np.exp(d[key][:,0]);ax.plot(d['z'][order,0],y[order],color=COLORS[r['seed']],label=f'seed {r["seed"]}')
            ax.set(title=title(n,m),xlim=(-zlimit,zlimit),ylim=ylims,xlabel='Fixed latent z',ylabel=ylabel);ax.legend(fontsize=8,ncol=2)
        finish(fig,figdir/(name+'.png'))
    # Basin-role probabilities for every fixed probe and every seed.
    fig,axs=subplots()
    for ax,(n,m) in zip(axs.flat,SIZES):
        mats=[]
        for r in groups[n,m]:
            d=r['diags'][20000];order=np.argsort(d['z'][:,0]);mats.append(d['basin_prob'][order].T)
        im=ax.imshow(np.vstack(mats),aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set(title=title(n,m),yticks=np.arange(12),yticklabels=[f's{s} {mode}' for s in range(4) for mode in ['L','C','R']],xlabel='2048 fixed latents (sorted by z)');ax.tick_params(axis='y',labelsize=8)
        for y in [2.5,5.5,8.5]:ax.axhline(y,color='white',lw=1)
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Conditional basin probability',shrink=.7);finish(fig,figdir/'06_latent_roles_all_seeds.png')
    fig,axs=subplots()
    for ax,(n,m) in zip(axs.flat,SIZES):
        for r in groups[n,m]:ax.plot([h['step'] for h in r['history']],[h['specialist_fraction'] for h in r['history']],color=COLORS[r['seed']],label=f'seed {r["seed"]}')
        ax.set(title=title(n,m),ylim=(-.03,1),xlabel='Training update',ylabel='Specialist fraction');ax.legend(fontsize=8,ncol=2)
    finish(fig,figdir/'07_role_emergence.png')
    # Frozen target vs finite teacher and policy mode masses; 4-seed mean and sample SD.
    fig,axs=subplots();truth=target_mass([-1,-.3,.3,1])
    for ax,(n,m) in zip(axs.flat,SIZES):
        rr=groups[n,m];teacher=np.stack([r['diags'][20000]['teacher_mode_mass'] for r in rr]);actor=np.stack([np.bincount(np.digitize(r['evals'][20000]['samples'],[-.3,.3]),minlength=3)/32768 for r in rr])
        for off,a,c,l in [(-.18,teacher,'#2076b6','Teacher'),(.18,actor,'#e06c36','Actor')]:ax.bar(np.arange(3)+off,a.mean(0),.34,yerr=a.std(0,ddof=1),capsize=3,color=c,label=l,alpha=.8)
        ax.plot(range(3),truth,'kd',label='Exact target');ax.set(title=title(n,m),xticks=range(3),xticklabels=['Left','Center','Right'],ylim=(0,.65),ylabel='Basin probability mass');ax.legend(fontsize=8,ncol=3)
    finish(fig,figdir/'08_teacher_actor_basin_mass.png')
    # Per-run parameter cosine is averaged, not gradients before taking cosine.
    fig,axs=subplots()
    for ax,(n,m) in zip(axs.flat,SIZES):
        a=np.mean([r['diags'][20000]['gradient_cosine'] for r in groups[n,m]],0);im=matrix(ax,a,1);ax.set(title=title(n,m)+' | mean of 4 seeds',yticklabels=['L','C','R'])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Gradient cosine');finish(fig,figdir/'09_gradient_cosine.png')
    fig,axs=plt.subplots(8,4,figsize=(15,22),constrained_layout=True)
    for i,(n,m) in enumerate(SIZES):
        for j,(key,normkey,label) in enumerate([('gradient_cosine','gradient_norm','All'),('cosine_trunk','norm_trunk','Shared trunk'),('cosine_mu_head','norm_mu_head','Mean head'),('cosine_sigma_head','norm_sigma_head','Sigma head')]):
            arr=[]
            for r in groups[n,m]:
                d=r['diags'][20000];a=d[key].copy();a[(d[normkey][:,None]*d[normkey][None,:])<=1e-30]=np.nan;arr.append(a)
            im=matrix(axs[i,j],mean_finite(arr),1);axs[i,j].set(title=f'{short(n,m)} | {label}',yticklabels=['L','C','R'])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.4,label='4-seed mean gradient cosine');finish(fig,figdir/'10_gradient_blocks.png')
    # Each row is a changed objective, each column is a measured mode. One common scale.
    for start,end,name in [(2,5,'11_mode_adam_reference'),(5,8,'12_mode_sgd_reference')]:
        mats=[np.mean([r['diags'][20000]['delta_reference_nll'][start:end] for r in groups[n,m]],0)*1000 for n,m in SIZES]
        lim=max(np.max(np.abs(x)) for x in mats);fig,axs=subplots()
        for ax,(n,m),mat in zip(axs.flat,SIZES,mats):
            im=matrix(ax,mat,lim);ax.set(title=title(n,m),yticklabels=['L update','C update','R update'],xlabel='Measured mode')
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Held-out NLL change × 1000 (positive = worse)');finish(fig,figdir/(name+'.png'))
    # Full vs momentum: individual seed points avoid misleading cancellation in averages.
    fig,axs=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for ax,space,desc in zip(axs,['delta_mode_nll','delta_reference_nll'],['Same training teacher / training latents','Held-out target / fixed evaluation latents']):
        for i,(n,m) in enumerate(SIZES):
            for r in groups[n,m]:
                d=r['diags'][20000];j=(r['seed']-1.5)*.025
                ax.scatter(i-.17+j,d[space][0].sum(),color=COLORS[r['seed']],marker='o',s=35)
                ax.scatter(i+.17+j,d[space][1].sum(),color=COLORS[r['seed']],marker='x',s=35)
        ax.axhline(0,c='black',lw=.6);ax.set_yscale('symlog',linthresh=1e-5);ax.set(xticks=range(8),xticklabels=[short(*x) for x in SIZES],title=desc,ylabel='Total NLL change: circle=full Adam, x=momentum only');ax.tick_params(axis='x',rotation=45)
    finish(fig,figdir/'13_full_vs_momentum.png')
    # Existing specialist groups stay fixed across the update; missing groups are not zero.
    for name,key,mult,desc in [('14_cross_mode_movement','cross_mode_position_rms',1000,'Representative action RMS move × 1000'),('15_cross_mode_mass','cross_mode_own_mass_change',1000,'Old specialist own-basin mass change × 1000')]:
        mats=[mean_finite([r['diags'][20000][key][2:5,:3] for r in groups[n,m]])*mult for n,m in SIZES]
        lim=max(np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0 for a in mats);fig,axs=subplots()
        for ax,(n,m),a in zip(axs.flat,SIZES,mats):
            im=matrix(ax,a,lim,positive=(key=='cross_mode_position_rms'));ax.set(title=title(n,m),yticklabels=['L update','C update','R update'],xlabel='Previously specialized group')
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label=desc);finish(fig,figdir/(name+'.png'))
    # Responsibilities are posterior mode assignments, not an OT map; show a successful and broad case for all seeds.
    fig,axs=plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
    for j,key in enumerate([(1024,1024),(2048,2048)]):
        for r in groups[key]:
            d=r['diags'][20000];ax=axs[r['seed'],j];order=np.argsort(d['training_z'][:,0]);im=ax.imshow(d['row_mode_fraction'][order].T,aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set(title=f'{short(*key)} | seed {r["seed"]}',yticks=range(3),yticklabels=['L','C','R'],xlabel='Fresh training latent (sorted by z)')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Responsibility-assigned mode fraction');finish(fig,figdir/'16_gmm_assignment.png')
    fig,axs=plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
    for j,key in enumerate([(1024,1024),(2048,2048)]):
        for r in groups[key]:
            d=r['diags'][20000];ax=axs[r['seed'],j];im=ax.imshow(d['row_mode_fraction'].T,aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set(title=f'{short(*key)} | seed {r["seed"]}',yticks=range(3),yticklabels=['L','C','R'],xlabel='Fresh training latent (original sampling order)')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Responsibility-assigned mode fraction');finish(fig,figdir/'18_gmm_assignment_raw.png')
    # Continuous identity across time. Select latent indices by z quantile, not by outcome.
    fig,axs=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
    for j,key in enumerate([(1024,1024),(2048,2048)]):
        for r in groups[key]:
            ax=axs[r['seed'],j];d=r['diags'][20000];order=np.argsort(d['z'][:,0]);ix=order[np.linspace(0,2047,64,dtype=int)];trace=np.stack([np.tanh(r['diags'][st]['mu'][ix,0]) for st in STEPS]);ax.plot(STEPS,trace,alpha=.5,lw=.65);ax.set(title=f'{short(*key)} | seed {r["seed"]}',ylim=(-1,1),xlabel='Training update',ylabel='Same-latent tanh(mu)')
    finish(fig,figdir/'17_fixed_latent_trajectories.png')

def stat(s,k):
    z=s[k];return 'N/A' if z['mean'] is None else f'{z["mean"]:.4f} ± {z["sd"]:.4f}' if z['sd'] is not None else f'{z["mean"]:.4f} (n={z["n"]})'
def make_tables(rows,summaries):
    a=['| N×M | Histogram TV ↓ | TV<0.1 | Specialist 비율 ↑ | 평균 σ |','|---|---:|---:|---:|---:|']
    b=['| N×M | Teacher basin TV ↓ | Teacher ESS/M | Component ESS/N | Teacher mode 누락 |','|---|---:|---:|---:|---:|']
    c=['| N×M | 음의 mode-gradient 쌍 | 단독 Adam: 다른 mode 악화 | 단독 SGD: 다른 mode 악화 | 다른/담당 latent 이동 비율 |','|---|---:|---:|---:|---:|']
    for s in summaries:
        a.append(f'| {short(s["n"],s["m"])} | {stat(s,"histogram_tv")} | {s["good_fit"]}/4 | {100*s["specialist_fraction"]["mean"]:.1f}% | {s["sigma_mean"]["mean"]:.3f} |')
        b.append(f'| {short(s["n"],s["m"])} | {stat(s,"teacher_basin_tv")} | {s["teacher_ess_ratio"]["mean"]:.3f} | {s["component_ess_ratio"]["mean"]:.3f} | {s["teacher_missing_mode_snapshots"]}/44 |')
        c.append(f'| {short(s["n"],s["m"])} | {100*s["pair_negative_fraction"]["mean"]:.1f}% | {100*s["mode_adam_offdiag_harm_fraction"]["mean"]:.1f}% | {100*s["mode_sgd_offdiag_harm_fraction"]["mean"]:.1f}% | {stat(s,"movement_ratio")} |')
    d=['| Run | Histogram TV | Specialist 비율 | Mean σ | 최초 TV<0.1 step |','|---|---:|---:|---:|---:|']
    for r in rows:d.append(f'| {r["run"]} | {r["histogram_tv"]:.4f} | {r["specialist_fraction"]:.3f} | {r["sigma_mean"]:.3f} | {r["first_histogram_tv_below_point1"] if r["first_histogram_tv_below_point1"] is not None else "도달 안 함"} |')
    return dict(RESULT_TABLE='\n'.join(a),TEACHER_TABLE='\n'.join(b),GRADIENT_TABLE='\n'.join(c),RUN_TABLE='\n'.join(d))

def html_report(md,out):
    # Render display formulas as offline PNGs; original LaTex is retained in report.md.
    def equation(match):
        tex=match.group(1).strip();key=hashlib.sha256(tex.encode()).hexdigest()[:12];path=out/'figures'/f'equation_{key}.png'
        fig=plt.figure(figsize=(12,.8));fig.text(.02,.5,'$'+tex+'$',fontsize=16,va='center');fig.savefig(path,dpi=170,bbox_inches='tight',pad_inches=.12);plt.close(fig)
        return '\n<img class="equation" alt="'+tex.replace('"','&quot;')+'" src="figures/'+path.name+'">\n'
    text=re.sub(r'\$\$(.*?)\$\$',equation,md,flags=re.S)
    body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
    for path in (out/'figures').glob('*.png'):body=body.replace('figures/'+path.name,'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())
    css='body{font:16px system-ui;max-width:1180px;margin:35px auto;padding:24px;line-height:1.7;color:#182936}h1{font-size:32px}h2{border-top:1px solid #dbe2e9;padding-top:28px;margin-top:40px}img{max-width:100%;height:auto}img.equation{max-height:100px;width:auto;display:block;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dbe2e9;padding:8px}th{background:#edf3f7}pre{overflow:auto;background:#edf3f7;padding:16px}blockquote{border-left:4px solid #2076b6;padding-left:16px}a{color:#087eaf}'
    (out/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Direct GMM: 32-run gradient interference</title><style>'+css+'</style></head><body>'+body+'</body></html>')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--analysis-commit',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);fig=a.out/'figures';fig.mkdir(exist_ok=True)
    runs,inputs,validation=collect(a.data);rows,summaries=analyze(runs)
    write(a.out/'per_run.json',rows);write(a.out/'summary.json',summaries);write(a.out/'INPUT_MANIFEST.json',dict(files=inputs,validation=validation,run_count=32,diagnostic_count=352,analysis_commit=a.analysis_commit))
    plot_all(runs,fig)
    md=(Path(__file__).parent/'report_template.md').read_text()
    for key,value in make_tables(rows,summaries).items():md=md.replace('{{'+key+'}}',value)
    md=md.replace('{{ANALYSIS_COMMIT}}',a.analysis_commit)
    assert '{{' not in md
    (a.out/'report.md').write_text(md);html_report(md,a.out)
    write(a.out/'REPORT_COMPLETE.json',dict(analysis_commit=a.analysis_commit,training_runs=32,updates=20000,diagnostics=352,figures=len(list(fig.glob('*.png')))))
    print('REPORT_COMPLETE',a.out)

if __name__=='__main__':main()
