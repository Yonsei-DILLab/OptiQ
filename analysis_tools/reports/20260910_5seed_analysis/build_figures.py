from pathlib import Path
import json, csv
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

P=Path(__file__).resolve().parent; F=P/'figures'; F.mkdir(exist_ok=True)
D=json.loads((P/'deep_analysis.json').read_text()); S=json.loads((P/'analysis_summary.json').read_text())
A=json.loads((P/'ot_audit_extended.json').read_text())
SUP=json.loads((P/'local_support_refined.json').read_text())
SEEDS=list(range(5)); C=['#bc4965','#2475ad','#e29236','#32927a','#795ba7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':145,'savefig.bbox':'tight'})
figures=[]
def save(fig,name):
    fig.savefig(F/(name+'.png'));fig.savefig(F/(name+'.pdf'));plt.close(fig);figures.append(name)
def vals(rows,key):return np.array([float(r[key]) for r in rows])
def fr(case,method,k=50,init='default'):
    return [r for r in D['frozen'] if r['case']==case and r['initialization']==init and r['method']==method and int(r['k'])==k and int(r['step'])==(0 if method.startswith('local_') else 20000)]
def ar(case,step=20000,init='default'):
    return sorted([r for r in A['rows'] if r['case']==case and r['initialization']==init and r['step']==step],key=lambda r:r['seed'])
cases=['unimodal','asymmetric_1.3_0.08','modes2d_4','modes2d_8','separable_4','separable_8']
labels=['1D single','1D asymmetric','2D / 4 modes','2D / 8 modes','4D separable','8D separable']

# All independent training seeds are visible; states remain nested in a seed.
fig,axs=plt.subplots(1,2,figsize=(11.2,4))
lm=['optiq_raw','optiq_td','local_no_is','local_official_is']; ll=['Raw actor','Actor + TD noise','Local no-IS','Local official-IS']
matrix=np.array([[next(r['mean_rmse'] for r in S['learned'] if r['name']==f'learned_seed{i}' and r['method']==m) for m in lm] for i in SEEDS])
for i,row in enumerate(matrix):axs[0].plot(range(4),row,'o-',lw=1,ms=5,color=C[i],alpha=.85,label=f'Seed {i}')
axs[0].set_xticks(range(4),ll,rotation=12);axs[0].set(ylabel='Mean state-wise RMSE',title='Same learned Q: K = 50, five seeds');axs[0].legend(fontsize=8)
for i,m in enumerate(['raw_bias','td_bias','local_bias']):
    for seed in SEEDS:
        rr=sorted([r for r in D['learned'] if r['seed']==seed and r['k']==50],key=lambda r:r['state'])
        axs[1].plot(vals(rr,'state'),vals(rr,m),color=C[i],alpha=.28,lw=.8)
    means=[];std=[]
    for state in sorted({r['state'] for r in D['learned']}):
        v=[r[m] for r in D['learned'] if r['k']==50 and r['state']==state];means.append(np.mean(v))
    axs[1].plot(sorted({r['state'] for r in D['learned']}),means,color=C[i],lw=2,label=ll[i])
axs[1].axhline(0,color='gray',ls=':',lw=1);axs[1].set(xlabel='MoveCar state x',ylabel='Bias: estimate mean − Boltzmann value',title='Thin lines: seeds; thick line: mean');axs[1].legend(fontsize=8)
fig.tight_layout();save(fig,'01_learned_backup')

fig,axs=plt.subplots(1,2,figsize=(12,4.2));xx=np.arange(6)
for i,case in enumerate(cases):
    truth=float(fr(case,'optiq_raw')[0]['truth']);axs[0].scatter(i,truth,marker='_',s=260,color='black',zorder=5,label='Reference' if i==0 else None)
    for j,(m,l) in enumerate(zip(['optiq_raw','optiq_td'],ll[:2])):
        r=fr(case,m);mu=vals(r,'mean');rm=vals(r,'rmse')
        axs[0].errorbar(i+(j-.5)*.20,mu.mean(),yerr=mu.std(ddof=1),fmt='o',color=C[j],capsize=3,label=l if i==0 else None)
        axs[1].errorbar(i+(j-.5)*.20,rm.mean(),yerr=rm.std(ddof=1),fmt='o',color=C[j],capsize=3,label=l if i==0 else None)
    local=sorted({r['method'] for r in D['frozen'] if r['case']==case and r['method'].startswith('local_no_is')})
    for ax,k in zip(axs,['mean','rmse']):
        v=[vals(fr(case,m),k).mean() for m in local];ax.plot([i+.26]*2,[min(v),max(v)],color=C[2],lw=4,alpha=.55,label='Local center range' if i==0 else None);ax.scatter([i+.26]*len(v),v,color=C[2],s=13)
for ax in axs:ax.set_xticks(xx,labels,rotation=18);ax.legend(fontsize=8)
axs[0].set(ylabel='Estimated backup mean',title='Frozen Q: target and estimate means');axs[1].set(ylabel='RMSE (log scale)',yscale='log',title='K = 50; error bars: seed SD')
fig.tight_layout();save(fig,'02_frozen_summary')

ks=[1,8,16,50,64,256,1024];fig,axs=plt.subplots(2,3,figsize=(12,7))
for ax,case,lab in zip(axs.flat,cases,labels):
    for m,col,l in [('optiq_raw',C[0],'Raw actor'),('optiq_td',C[1],'Actor + TD noise'),('reference',C[3],'Reference samples')]:
        ys=np.array([vals(fr(case,m,k),'rmse') for k in ks]);ax.plot(ks,ys.mean(1),'o-',ms=3,color=col,label=l);ax.fill_between(ks,ys.min(1),ys.max(1),color=col,alpha=.1)
    methods=sorted({r['method'] for r in D['frozen'] if r['case']==case and r['method'].startswith('local_no_is')})
    yy=np.array([[vals(fr(case,m,k),'rmse').mean() for k in ks] for m in methods]);ax.fill_between(ks,yy.min(0),yy.max(0),color=C[2],alpha=.25,label='Local center range');ax.plot(ks,yy.T,color=C[2],lw=.7,alpha=.7)
    ax.set(xscale='log',yscale='log',title=lab,xlabel='Actions per estimate K',ylabel='Backup RMSE');ax.grid(alpha=.13)
axs[0,0].legend(fontsize=7);fig.suptitle('More samples reduce Monte Carlo variance; persistent bias remains',y=1.01);fig.tight_layout();save(fig,'03_rmse_vs_k')

fig,axs=plt.subplots(1,4,figsize=(12.5,3.6));steps=[0,100,1000,5000,20000]
for ax,case,lab in zip(axs,cases[2:],labels[2:]):
    for init,col,l in [('default',C[0],'Default'),('coverage',C[1],'Coverage warm start')]:
        y=np.array([[next(r['bin_tv'] for r in D['mode_history'] if r['case']==case and r['initialization']==init and r['seed']==i and r['step']==st) for st in steps] for i in SEEDS])
        for row in y:ax.plot(range(5),row,color=col,lw=.6,alpha=.35)
        ax.plot(range(5),y.mean(0),'o-',color=col,lw=2,ms=4,label=l)
    ax.set_xticks(range(5),['0','100','1k','5k','20k'],rotation=25);ax.set(ylim=(-.03,1.03),title=lab,xlabel='OT updates (categorical spacing)');ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Mode-bin discrepancy ½ Σ |p − p*|');axs[1].legend(fontsize=7);fig.tight_layout();save(fig,'04_mode_history')

fig,axs=plt.subplots(2,4,figsize=(12.4,6.3))
for i,case in enumerate(['modes2d_4','modes2d_8']):
    ref=np.load(P/'runs'/f'frozen_{case}_default_seed0'/'reference.npz');edges=np.linspace(-1,1,101)
    reference=np.histogram2d(*ref['grid'].T,bins=edges,weights=ref['mass'])[0]
    hs=[reference]
    for init,step in [('default',20000),('coverage',0),('coverage',20000)]:
        a=np.load(P/'runs'/f'frozen_{case}_{init}_seed0'/f'distribution_{step}.npz')['samples'];h=np.histogram2d(*a.T,bins=edges)[0];hs.append(h/h.sum())
    norm=PowerNorm(.35,vmin=0,vmax=max(h.max() for h in hs))
    for j,(h,lab) in enumerate(zip(hs,['Target','Default: 20k','Coverage: 0','Coverage: 20k'])):
        axs[i,j].imshow(h.T,origin='lower',extent=(-1,1,-1,1),cmap='magma',norm=norm);axs[i,j].set(title=lab,xlabel='Action 1');axs[i,j].set_aspect('equal')
    axs[i,0].set_ylabel(f'{4 if i==0 else 8} modes · Action 2')
fig.suptitle('Seed 0 illustrations: matching mode masses does not imply matching density',y=1.01);fig.tight_layout();save(fig,'05_density')

fig,axs=plt.subplots(1,3,figsize=(13,4))
r=ar('modes2d_8');keys=['weighted','row_expected','hard','long_row_expected','long_hard'];labs=['Weighted\ncandidates','Row sampling\nexpectation (30)','Argmax\ntarget (30)','Row sampling\nexpectation (300)','Argmax\ntarget (300)']
for seed,row in enumerate(r):axs[0].plot(range(5),[row['q_mean'][k] for k in keys],'o-',color=C[seed],alpha=.7,lw=.8,ms=3)
axs[0].axhline(float(fr('modes2d_8','optiq_raw')[0]['truth']),color='black',ls='--',label='Global Boltzmann value');axs[0].set_xticks(range(5),labs,rotation=32,ha='right');axs[0].set(ylabel='Q mean of target actions',title='2D / 8 modes: target selection');axs[0].legend(fontsize=7)
for j,case in enumerate(['modes2d_4','modes2d_8']):
    rr=ar(case)
    for seed,row in enumerate(rr):axs[1].plot([j-.17,j+.17],[row['telemetry']['row_l1'],row['telemetry']['long_row_l1']],color=C[j],alpha=.5,lw=.8,marker='o',ms=4)
axs[1].set_xticks([-.17,.17,.83,1.17],['4 modes\n30 iter.','4 modes\n300 iter.','8 modes\n30 iter.','8 modes\n300 iter.']);axs[1].set(yscale='log',ylabel='Row marginal L1 error',title='More iterations fix row constraints')
for init,col in [('default',C[0]),('coverage',C[1])]:
    for seed in SEEDS:
        y=[next(r['jacobian_min_max_ratio'] for r in ar('modes2d_8',step,init) if r['seed']==seed) for step in [0,20000]];axs[2].plot([0,1],y,'o-',color=col,alpha=.55,lw=1,ms=4,label=init if seed==0 else None)
axs[2].set_xticks([0,1],['Initial','20k']);axs[2].set(yscale='log',ylabel='Median Jacobian σmin / σmax',title='2D actor: learned anisotropy');axs[2].legend(fontsize=8)
fig.tight_layout();save(fig,'06_ot_audit')

fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,case in zip(axs,['modes2d_8','separable_8']):
    rr=[r for r in SUP if r['case']==case];xx=np.arange(len(rr));truth=rr[0]['global_truth']
    ax.axhline(truth,color='black',ls='--',label='Global value');ax.scatter(xx,[r['conditional_truth'] for r in rr],marker='D',color=C[3],label='Correct IS limit on support',s=35)
    for j,(m,col,l) in enumerate([('truncated_is',C[1],'Correct truncated IS'),('no_is',C[2],'No IS'),('official_is',C[4],'Official-style IS')]):
        means=[vals(fr(case,f'local_{m}_center{r["center"]}',1024),'mean').mean() for r in rr];ax.scatter(xx+(j-1)*.14,means,color=col,s=26,label=l)
    ax.set_xticks(xx,[f'C{r["center"]}\n{r["target_support_mass"]*100:.2f}%' for r in rr]);ax.set(title=case.replace('modes2d_8','2D / 8 modes').replace('separable_8','8D separable'),ylabel='Backup estimate mean',xlabel='Proposal center / target mass within support')
axs[0].legend(fontsize=7);fig.suptitle('K = 1,024: correcting proposal density does not recover missing support',y=1.01);fig.tight_layout();save(fig,'07_local_support')

fig,axs=plt.subplots(1,3,figsize=(12.5,3.8));arms=['max','boltzmann','actor','local'];al=['Grid max','Grid Boltzmann','Actor (K=1)','Local (K=50)']
for seed in SEEDS:
    r=[next(r for r in D['controls'] if r['arm']==a and r['seed']==seed) for a in arms]
    axs[0].plot(range(4),[v['barrier'] for v in r],'o-',color=C[seed],ms=5,lw=.9,label=f'Seed {seed}')
    axs[1].plot(range(4),[v['perturbation_second_difference'] for v in r],'o-',color=C[seed],ms=5,lw=.9)
    axs[2].plot(range(4),[v['grid_error'] for v in r],'o-',color=C[seed],ms=4,lw=.9)
for ax in axs:ax.set_xticks(range(4),al,rotation=20)
axs[0].set(ylabel='Barrier above endpoint objectives',title='Common replay: landscape barrier');axs[0].legend(fontsize=7)
axs[1].set(ylabel='Median |L(θ+δ)+L(θ−δ)−2L(θ)|',title='Local second difference (radius 0.05)')
axs[2].axhline(.001,color='black',ls='--',label='Gate: 1e−3');axs[2].set(yscale='log',ylabel='Max grid-to-reference error',title='All 20 corrected controls pass');axs[2].legend(fontsize=7)
fig.tight_layout();save(fig,'08_control')

fig,axs=plt.subplots(2,5,figsize=(13,5.6))
for i,(family,methods) in enumerate([('cross_ddpg_sd2',['ddpg','sd2']),('cross_td3_sd3',['td3','sd3'])]):
    for seed in SEEDS:
        ax=axs[i,seed];z=np.load(P/'runs'/f'{family}_seed{seed}'/'landscape.npz')
        for j,m in enumerate(methods):
            y=z[m+'_path'];ax.plot(z[m+'_alpha'],y-y[0],color=[C[1],C[2]][j],lw=1.6,label=m.upper()+' critic')
        ax.set(title=f'Seed {seed}',xlabel='Actor interpolation α');ax.legend(fontsize=7)
    axs[i,0].set_ylabel(f'{methods[0].upper()} → {methods[1].upper()}\nL(α) − L(0)')
fig.suptitle('Same actor path, two critics — both endpoints achieve return 188',y=1.01);fig.tight_layout();save(fig,'09_cross_landscape')

fig,axs=plt.subplots(1,2,figsize=(11.5,3.7));methods=['ddpg','sd2','td3','sd3','optiq']
for m,col in zip(methods,[C[1],C[2],C[3],C[4],C[0]]):
    curves=[json.loads((P/'runs'/f'movecar_{m}_seed{i}'/'learning.json').read_text()) for i in SEEDS];x=np.array([r['step'] for r in curves[0]]);y=np.array([[r['return_mean'] for r in c] for c in curves]);axs[0].plot(x,y.mean(0),color=col,label=m.upper());axs[0].fill_between(x,y.min(0),y.max(0),color=col,alpha=.13)
axs[0].set(xlabel='Environment steps',ylabel='100-step undiscounted return',ylim=(0,200),title='MoveCar: mean and seed range');axs[0].legend(fontsize=8,ncol=2)
for i,m in enumerate(methods):
    r=sorted([r for r in S['movecar'] if r['method']==m],key=lambda r:r['seed']);axs[1].scatter(np.linspace(i-.12,i+.12,5),[r['return_mean'] for r in r],color=C[0] if m=='optiq' else C[1],s=28)
axs[1].axhline(188,color='gray',ls=':');axs[1].set_xticks(range(5),[m.upper() for m in methods]);axs[1].set(ylim=(178,190),ylabel='Final return',title='Each point is one training seed')
fig.tight_layout();save(fig,'10_movecar')

fig,axs=plt.subplots(1,5,figsize=(13,3.2))
for seed,ax in enumerate(axs):
    z=np.load(P/'runs'/f'cross_optiq_optiq_seed{seed}'/'landscape.npz')
    for m,col in zip(methods,[C[1],C[2],C[3],C[4],C[0]]):
        y=z[m+'_path'];ax.plot(z[m+'_alpha'],y-y[0],color=col,label=m.upper(),lw=1.1)
    ax.set(title=f'Seed {seed}',xlabel='OptiQ actor interpolation α')
axs[0].set_ylabel('L(α) − L(0)');axs[4].legend(fontsize=7);fig.suptitle('Same OptiQ early → final actor path under five different critics',y=1.03);fig.tight_layout();save(fig,'11_optiq_cross')

counts=Counter(d.name.split('_')[0] for d in (P/'runs').iterdir() if d.is_dir())
assert dict(counts)==dict(frozen=130,movecar=25,control=20,landscape=25,learned=5,cross=15),counts
for d in (P/'runs').iterdir():
    if d.is_dir():assert int(d.name.rsplit('seed',1)[1]) in SEEDS and (d/'COMPLETE').exists(),d
assert len(A['rows'])==80 and all(r['input_latent_dim']==r['output_dim'] for r in A['rows'])
assert max(r['grid_error'] for r in D['controls'])<.001
(P/'figure_manifest.json').write_text(json.dumps({'figures':figures,'completed_run_counts':dict(counts),'seeds':SEEDS},indent=2))
print('Generated and validated',len(figures),'PNG/PDF figure pairs;',sum(counts.values()),'completed runs.')
