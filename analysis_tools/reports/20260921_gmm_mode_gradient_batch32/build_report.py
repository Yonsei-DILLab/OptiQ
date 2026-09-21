"""Completed-run-only batch comparison; no numerical training code is modified."""
import argparse, base64, datetime, html, json, math, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--studies',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
OUT=a.output;OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
METHODS=['baseline','mode_only','mode_confidence'];LABELS=['Direct GMM','Mode selection','Selection + confidence'];COLORS=['#07889b','#8554ae','#e36b35']
SIZES=[(16,16),(64,64),(128,128),(256,256),(1024,1024),(2048,2048),(64,4096),(2048,4096)]
ROOTS={1:a.studies/'20260921_gmm_mode_gradient',32:a.studies/'20260921_gmm_mode_gradient_batch32'}
STAMP=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
records={}; progress=[]; manifests={}; arrays={}; series={}
for B,study in ROOTS.items():
 manifest=json.loads((study/'campaign/SOURCE_MANIFEST.json').read_text());manifests[B]=manifest['commit']
 for run in sorted((study/'campaign/runtime/runs').glob('*')):
  if not run.is_dir() or not (run/'config.json').exists():continue
  cfg=json.loads((run/'config.json').read_text());key=(B,cfg['n'],cfg['m'],cfg['method'],cfg['seed'])
  state=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
  if not (run/'COMPLETE.json').exists():progress.append(dict(batch=B,name=run.name,step=state.get('step',0),failed=(run/'FAILED.json').exists()));continue
  completion=json.loads((run/'COMPLETE.json').read_text())
  if completion['step']!=20000 or not (run/'eval_020000.npz').exists():raise RuntimeError('Invalid COMPLETE '+str(run))
  with np.load(run/'eval_020000.npz') as d: ev={k:d[k] for k in d.files}
  assert len(ev['samples'])==32768 and len(ev['histogram'])==256
  np.testing.assert_allclose(ev['histogram'],np.histogram(ev['samples'],ev['edges'])[0]/32768,atol=1e-12)
  np.testing.assert_allclose(ev['target_bin_mass'].sum(),1,atol=2e-7)
  history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()];last=history[-1]
  tv=float(.5*np.abs(ev['histogram']-ev['target_bin_mass']).sum());assert abs(tv-last['histogram_tv'])<1e-8
  diagfile=run/'diagnostic_020000.npz'
  with np.load(diagfile) as d:
   keep=['H','training_z','gradient_gram','gradient_cosine','gradient_norm','batch_H','batch_teacher_mode_mass','routed_gradient_norm','routed_batch_metrics','single_group_gradient_cosine','teacher_mode_mass']
   dg={k:d[k] for k in keep if k in d}
  logs=[json.loads(x) for x in (run/'training.jsonl').read_text().splitlines()]
  launch=json.loads((run/'LAUNCH.json').read_text()) if (run/'LAUNCH.json').exists() else {}
  counts=ev['specialist_counts'].tolist(); mass=np.bincount(np.digitize(ev['samples'],[-.3,.3]),minlength=3)/32768
  rec=dict(batch=B,n=cfg['n'],m=cfg['m'],method=cfg['method'],seed=cfg['seed'],step=20000,tv=tv,specialist_fraction=last['specialist_fraction'],sigma_mean=last['sigma_mean'],specialist_counts=counts,mode_mass=mass.tolist(),train_seconds=completion['train_seconds'],diagnostic_seconds=completion['diagnostic_seconds'],hostname=launch.get('hostname'),commit=cfg['source_commit'],run_path=str(run),last_training_metrics=logs[-1]['last_metrics'])
  records[key]=rec;arrays[key]=(ev,dg);series[key]=history
COUNTS={B:sum(k[0]==B for k in records) for B in ROOTS};FINAL=COUNTS[32]==96
PAIR={}
for n,m in SIZES:
 for method in METHODS:
  ss=[s for s in range(4) if all((B,n,m,method,s) in records for B in (1,32))]
  if ss:PAIR[n,m,method]=ss
COMMON={size:[s for s in range(4) if all((B,*size,method,s) in records for B in (1,32) for method in METHODS)] for size in SIZES}
COMMON={k:v for k,v in COMMON.items() if v}
SHOW=list(COMMON)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(name,fig):fig.savefig(OUT/'figures'/name,dpi=150,bbox_inches='tight');plt.close(fig)
def target(x):
 norm=sum((math.erf((1-c)/(.1*np.sqrt(2)))-math.erf((-1-c)/(.1*np.sqrt(2))))/6 for c in [-.6,0,.6])
 return sum(np.exp(-.5*((x-c)/.1)**2) for c in [-.6,0,.6])/(3*.1*np.sqrt(2*np.pi)*norm)
# Completed common seeds only; averaging raw histograms is pooling, never KDE.
fig,axs=plt.subplots(2,max(1,len(SHOW)),figsize=(4.1*max(1,len(SHOW)),7),squeeze=False,sharey=True)
for col,(n,m) in enumerate(SHOW):
 ss=COMMON[n,m]
 for row,B in enumerate((1,32)):
  ax=axs[row,col];x=np.linspace(-1,1,1400);ax.plot(x,target(x),'--',c='#24334a',lw=1.8,label='Exact target')
  for meth,label,c in zip(METHODS,LABELS,COLORS):
   evs=[arrays[B,n,m,meth,s][0] for s in ss];edges=evs[0]['edges'];den=np.array([e['histogram']/np.diff(edges) for e in evs]);mid=(edges[1:]+edges[:-1])/2
   ax.stairs(den.mean(0),edges,color=c,lw=1.25,label=label)
   if len(ss)>1:ax.fill_between(mid,den.min(0),den.max(0),color=c,alpha=.08)
  ax.set(title=f'Batch {B} | {n} x {m}\n20K; paired seeds {ss}',xlabel='Action',ylabel='Density');ax.set_xlim(-1,1);ax.grid(alpha=.15)
  if col==0 and row==0:ax.legend(fontsize=8)
fig.suptitle('Batch 1 vs 32 — 32,768 sampled actions / run; raw 256-bin histograms',fontsize=15);fig.tight_layout();save('density_comparison.png',fig)
# All batch-1 evidence, no seed-zero-only conclusion.
fig,axs=plt.subplots(4,2,figsize=(13,13),sharex=True)
for ax,(n,m) in zip(axs.flat,SIZES):
 x=np.linspace(-1,1,1000);ax.plot(x,target(x),'--',c='#24334a',lw=1.5,label='Target')
 for meth,label,c in zip(METHODS,LABELS,COLORS):
  for s in range(4):
   e=arrays[1,n,m,meth,s][0];ax.stairs(e['histogram']/np.diff(e['edges']),e['edges'],color=c,alpha=.45,lw=.65,label=label if s==0 else None)
 ax.set(title=f'{n} x {m} | batch 1, all 4 seeds, 20K',xlabel='Action',ylabel='Density');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8);fig.tight_layout();save('batch1_all_seeds.png',fig)
fig,axs=plt.subplots(max(1,len(SHOW)),2,figsize=(12,3.2*max(1,len(SHOW))),squeeze=False)
for row,(n,m) in enumerate(SHOW):
 ss=COMMON[n,m]
 for meth,label,c in zip(METHODS,LABELS,COLORS):
  for B,style in [(1,'--'),(32,'-')]:
   for si,s in enumerate(ss):
    h=series[B,n,m,meth,s]
    for col,xkey in enumerate(('step','train_seconds')):
     axs[row,col].plot([v[xkey] for v in h],[v['histogram_tv'] for v in h],style,c=c,lw=1.2,alpha=.8,label=f'{label}, B={B}' if si==0 else None)
 for col in (0,1):axs[row,col].set(title=f'{n} x {m} | paired seeds {ss}',xlabel='Optimizer updates' if col==0 else 'Recorded training seconds (incl. compile)',ylabel='Histogram TV',ylim=(0,.55));axs[row,col].grid(alpha=.15)
axs[0,0].legend(fontsize=7,ncol=2);fig.tight_layout();save('tracking_steps_time.png',fig)
# One representative setting selected by fixed preference, not by best outcome.
REP=next((s for s in [(128,128),(64,64),(16,16)] if s in COMMON),SHOW[0]);seed=COMMON[REP][0];n,m=REP
for sorted_z in (False,True):
 fig,axs=plt.subplots(2,3,figsize=(14,5.4),squeeze=False)
 for row,B in enumerate((1,32)):
  for col,(method,label) in enumerate(zip(METHODS,LABELS)):
   _,d=arrays[B,n,m,method,seed];order=np.argsort(d['training_z'][:,0]) if sorted_z else np.arange(n);ax=axs[row,col]
   im=ax.imshow(d['H'][order].T,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='viridis')
   ax.set(title=f'{label} | B={B}',xlabel='Latent z rank' if sorted_z else 'Original sample index',yticks=[0,1,2],yticklabels=['Left','Center','Right'])
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='H(mode | sampled component)');fig.suptitle(f'Responsibility: {n} x {m}, seed {seed}, 20K; B32 shows group 0')
 save('responsibility_sorted.png' if sorted_z else 'responsibility_raw.png',fig)
fig,axs=plt.subplots(2,3,figsize=(14,6),sharex=True)
for col,(method,label,c) in enumerate(zip(METHODS,LABELS,COLORS)):
 for B,style in [(1,'--'),(32,'-')]:
  e,_=arrays[B,n,m,method,seed];order=np.argsort(e['z'][:,0]);z=e['z'][order,0]
  axs[0,col].plot(z,np.tanh(e['mu'][order,0]),style,label=f'B={B}',c='#66758a' if B==1 else c)
  axs[1,col].plot(z,np.exp(e['log_sigma'][order,0]),style,label=f'B={B}',c='#66758a' if B==1 else c)
 for level in [-.6,0,.6]:axs[0,col].axhline(level,c='k',ls=':',alpha=.3)
 axs[0,col].set(title=label,ylabel='tanh(mu(z))');axs[0,col].legend();axs[1,col].set(xlabel='Fixed evaluation latent z',ylabel='Pre-tanh sigma(z)')
 for row in (0,1):axs[row,col].grid(alpha=.15)
fig.suptitle(f'Fixed latent bank: {n} x {m}, seed {seed}, 20K (tanh(mu) is not E[action])');fig.tight_layout();save('latent_specialization.png',fig)
fig,axs=plt.subplots(2,3,figsize=(12,7))
for row,B in enumerate((1,32)):
 for col,(method,label) in enumerate(zip(METHODS,LABELS)):
  _,d=arrays[B,n,m,method,seed];c=d['gradient_cosine'];ax=axs[row,col];im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=10)
  ax.set(title=f'{label} | B={B}',xticks=range(3),yticks=range(3),xticklabels=['L','C','R'],yticklabels=['L','C','R'])
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='Cosine');fig.suptitle(f'Original marginal-NLL mode gradients | {n} x {m}, seed {seed}, 20K\nB1: one group; B32: averaged 32 groups; actors/teachers differ between panels')
save('gradient_cosine.png',fig)
# Numeric summary retains every completed seed, not just illustrative panels.
comp=[]
for (nn,mm,method),ss in PAIR.items():
 v1=np.array([records[1,nn,mm,method,s]['tv'] for s in ss]);v32=np.array([records[32,nn,mm,method,s]['tv'] for s in ss])
 comp.append(dict(n=nn,m=mm,method=method,seeds=ss,batch1_tv_mean=float(v1.mean()),batch32_tv_mean=float(v32.mean()),paired_delta=float((v32-v1).mean()),batch1_tv_std=float(v1.std(ddof=1)) if len(ss)>1 else None,batch32_tv_std=float(v32.std(ddof=1)) if len(ss)>1 else None))
summary=dict(snapshot=STAMP,final=FINAL,completed=COUNTS,expected_per_batch=96,commits=manifests,paired=comp,runs=list(records.values()),incomplete=progress,representative=dict(n=n,m=m,seed=seed),data_validation='Recomputed all reported TVs from 32768 actual samples; exact target bin masses sum to 1.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(OUT/'paired_console.json').write_text(json.dumps(dict(snapshot=STAMP,counts=COUNTS,paired=comp),indent=2)+'\n')
# Write a compact content skeleton; interpretation is supplied in a separate verified note.
rows=['| N×M | 방법 | 비교 seed | batch 1 TV | batch 32 TV | 차이 (32−1) |','|---|---|---|---:|---:|---:|']
for r in comp:
 label=LABELS[METHODS.index(r['method'])];rows.append(f"| {r['n']}×{r['m']} | {label} | {','.join(map(str,r['seeds']))} | {r['batch1_tv_mean']:.4f} | {r['batch32_tv_mean']:.4f} | {r['paired_delta']:+.4f} |")
(OUT/'paired_table.md').write_text('\n'.join(rows)+'\n')
rows=['| Batch | N×M | 방법 | Seed | TV | Specialist 비율 | 평균 sigma | Training 초 | Diagnostic 초 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for k,r in sorted(records.items()):rows.append(f"| {r['batch']} | {r['n']}×{r['m']} | {r['method']} | {r['seed']} | {r['tv']:.4f} | {r['specialist_fraction']:.3f} | {r['sigma_mean']:.4f} | {r['train_seconds']:.1f} | {r['diagnostic_seconds']:.1f} |")
(OUT/'all_runs_table.md').write_text('\n'.join(rows)+'\n')
print(json.dumps(dict(snapshot=STAMP,counts=COUNTS,representative=summary['representative'],paired=comp),indent=2),flush=True)
