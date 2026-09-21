"""Self-contained MD/HTML report; incomplete runs are never called final results."""
from pathlib import Path
import argparse,json,html,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment.config import METHODS,SIZES
from experiment.problems import GRID,EDGES,schedule,analytic_q,analytic_cdf,reward

LABELS={'gmm_learned':'Direct GMM (learned sigma)','exact_learned':'Exact OT (learned sigma)',
 'sinkhorn_learned':'Sinkhorn (learned sigma)','exact_fixed05':'Exact OT (sigma=.5)',
 'sinkhorn_fixed05':'Sinkhorn (sigma=.5)','exact_fixed01':'Exact OT (sigma=.1)',
 'sinkhorn_fixed01':'Sinkhorn (sigma=.1)'}
STAGES=['mass','split','replay','closed']
COLORS=['#e4572e','#7657a8','#008c9d','#c28b16','#457b9d','#4a8c47','#cc6688']

def finish(fig,path):
 fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
  *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])
def paths_for(root,t):
 own=root/'runs'/t['name'];p=[]
 if t.get('parent'):p.append(root/'runs'/t['parent'])
 return [*p,own]
def densities(root,t):
 data={}
 for d in paths_for(root,t):
  progress=d/'progress.json';limit=json.loads(progress.read_text())['step'] if progress.exists() else 0
  for f in sorted((d/'density').glob('[0-9]*.npz')):
   n=int(f.stem)
   if n<=limit:data[n]=f
 return data
def records(root,t):
 rows={}
 for d in paths_for(root,t):
  progress=d/'progress.json';limit=json.loads(progress.read_text())['step'] if progress.exists() else 0
  for f in sorted((d/'evaluations').glob('*.npz')):
   with np.load(f) as z:
    for i,s in enumerate(z['step']):
     if s<=limit:rows[int(s)]={k:float(z[k][i]) for k in z.files}
 return [rows[k] for k in sorted(rows)]
def shrink(a):
 # Preserve all mass-bearing entries: contiguous block averages, never stride sampling.
 r=np.linspace(0,a.shape[0],min(a.shape[0],256)+1,dtype=int)
 c=np.linspace(0,a.shape[1],min(a.shape[1],512)+1,dtype=int)
 v=np.add.reduceat(np.add.reduceat(a,r[:-1],axis=0),c[:-1],axis=1)
 return v/np.diff(r)[:,None]/np.diff(c)[None]

def build(root,out):
 out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 ts=json.loads((root/'tasks.json').read_text());manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
 # Planned Q figures are available before any training result.
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for col,stage in enumerate(['mass','split']):
  snapshots=[20000,20001,25001,30001] if stage=='mass' else [20000,21000,22000,27000]
  for t in snapshots:
   p=schedule(stage,t);q=analytic_q(GRID,p);c,w=p.astype(float)
   from scipy.special import ndtr
   z=np.sum(w*(ndtr((1-c)/.1)-ndtr((-1-c)/.1)));pdf=np.exp(q/.25)/z
   axes[0,col].plot(GRID,q,label=f'update {t:,}');axes[1,col].plot(GRID,pdf)
  axes[0,col].set(title=stage+' | Q(a)',ylabel='Q');axes[0,col].legend(fontsize=8)
  axes[1,col].set(xlabel='Action',ylabel='Exact target density')
 finish(fig,figdir/'00_q_schedules.png')
 fig,ax=plt.subplots(figsize=(9,3))
 for s in [-.6,0,.6]:ax.plot(GRID,reward(s,GRID),label=f's={s}')
 ax.set(xlabel='Next position / action',ylabel='Reward',title='Stationary MDP: next state = action');ax.legend()
 finish(fig,figdir/'01_mdp_reward.png')
 counts={s:0 for s in ['prefix',*STAGES,'source']};running=0;failed=[];summaries=[]
 for t in ts:
  d=root/'runs'/t['name'];done=(d/'COMPLETE.json').exists()
  if done:counts[t['stage']]+=1
  elif (d/'progress.json').exists():running+=1
  if (d/'FAILED.json').exists() and not done:failed.append(t['name'])
  if done and t['stage'] in STAGES:
   with np.load(d/'density/final_independent.npz') as z:
    row={k:t[k] for k in ('stage','method','n','m','seed')}
    for k in ['binned_tv','basin_mass_tv','backup_bias','teacher_binned_tv','actor_teacher_binned_tv','common_marginal_nll','sigma_mean','between_mu_variance','within_variance']:
     row[k]=float(z[k])
   ev=records(root,t)
   row['stochastic_return_mean']=ev[-1].get('stochastic_return_mean','') if ev else ''
   e=[r for r in ev if r['step']>=20000] if t['stage'] in ('mass','split') else ev
   row['tracking_TV_AUC']=float(np.trapezoid([r['binned_tv'] for r in e],[r['step'] for r in e])) if hasattr(np,'trapezoid') else float(np.trapz([r['binned_tv'] for r in e],[r['step'] for r in e]))
   row['recovery_updates']='';row['recovery_censored']=''
   if t['stage'] in ('mass','split'):
    baseline=np.mean([r['binned_tv'] for r in ev if 19000<=r['step']<=20000]);at=30000 if t['stage']=='mass' else 27000
    post=[r for r in ev if r['step']>at];hits=0;recovery=None
    for r in post:
     hits=hits+1 if r['binned_tv']<=baseline+.05 else 0
     if hits==3:recovery=r['step']-at;break
    row['recovery_updates']=recovery if recovery is not None else '';row['recovery_censored']=recovery is None
   summaries.append(row)
 if summaries:
  with (out/'summary.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 images=[]
 for stage in STAGES:
  fig,axes=plt.subplots(7,4,figsize=(18,21));has=False
  for i,method in enumerate(METHODS):
   for j,(n,m) in enumerate(SIZES):
    ax=axes[i,j];selected=[t for t in ts if t['stage']==stage and t['method']==method and t['n']==n]
    full=[root/'runs'/t['name']/'density/final_independent.npz' for t in selected if (root/'runs'/t['name']/'COMPLETE.json').exists()]
    ax.set_title(f'{LABELS[method]}\n{n}x{m}; complete seeds={len(full)}/4',fontsize=9)
    if full:
     ds=[np.load(f) for f in full];p=np.stack([d['density'] for d in ds]);ref=np.stack([d['target_bin_mass']/np.diff(EDGES) for d in ds]);x=(EDGES[:-1]+EDGES[1:])/2
     mean=p.mean(0);sd=p.std(0,ddof=1) if len(p)>1 else np.zeros_like(mean)
     ax.plot(x,ref.mean(0),'--',color='#243248',label='Target for these runs');ax.plot(x,mean,color=COLORS[i],label='Sample histogram')
     ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=COLORS[i],alpha=.2);has=True
     for d in ds:d.close()
    else:ax.text(.5,.5,'Not completed',transform=ax.transAxes,ha='center')
    if i==6:ax.set_xlabel('Action')
    if j==0:ax.set_ylabel('Density')
  if has:
   name=f'10_final_{stage}.png';finish(fig,figdir/name);images.append((stage,name))
  else:plt.close(fig)
 # Prespecified seed 0 / N16 trajectories; targets are paired with each method.
 for stage in STAGES:
  selected=[t for t in ts if t['stage']==stage and t['n']==16 and t['seed']==0]
  available=[(t,densities(root,t)) for t in selected];available=[(t,d) for t,d in available if len(d)>2]
  if not available:continue
  fig,axes=plt.subplots(len(available),2,figsize=(13,2.6*len(available)),squeeze=False)
  for i,(t,ds) in enumerate(available):
   steps=np.array(sorted(ds));z=[np.load(ds[k]) for k in steps];actor=np.stack([d['density'] for d in z]);target=np.stack([d['target_bin_mass']/np.diff(EDGES) for d in z])
   vmax=max(actor.max(),target.max());edges_t=np.r_[steps[0],(steps[:-1]+steps[1:])/2,steps[-1]+1]
   for j,arr in enumerate([target,actor]):
    im=axes[i,j].pcolormesh(EDGES,edges_t,arr,cmap='magma',vmin=0,vmax=vmax,shading='flat')
    axes[i,j].set(title=LABELS[t['method']]+(' | target' if j==0 else ' | actual samples'),xlabel='Action',ylabel='Actor update');fig.colorbar(im,ax=axes[i,j],label='Density')
   for d in z:d.close()
  name=f'20_tracking_{stage}.png';finish(fig,figdir/name);images.append((stage+' tracking (seed 0, N16)',name))
 # One current assignment per method, predetermined small-size seed.
 small=[t for t in ts if t['stage']=='mass' and t['n']==16 and t['seed']==0]
 have=[]
 for t in small:
  ps=sorted((root/'runs'/t['name']/'assignments').glob('[0-9]*.npz'))
  if not ps and t.get('parent'):ps=sorted((root/'runs'/t['parent']/'assignments').glob('[0-9]*.npz'))
  if ps:have.append((t,ps[-1]))
 if have:
  fig,axes=plt.subplots(len(have),2,figsize=(13,2.8*len(have)),squeeze=False)
  for i,(t,p) in enumerate(have):
   with np.load(p) as z:
    A=z['A'][0];rs=z['source_order'];cs=z['candidate_order'];u=shrink(A);v=shrink(A[rs][:,cs]);vmax=max(u.max(),v.max())
    for j,d in enumerate([u,v]):
     im=axes[i,j].imshow(d,aspect='auto',origin='lower',interpolation='nearest',vmin=0,vmax=vmax,cmap='magma')
     axes[i,j].set(title=f'{LABELS[t["method"]]}, update {int(z["step"]):,} | '+('original order' if j==0 else 'action sorted'),xlabel='Teacher columns',ylabel='Student rows')
     fig.colorbar(im,ax=axes[i,j],label='Mean joint mass / entry')
  name='30_assignment_original_sorted.png';finish(fig,figdir/name);images.append(('Assignment (seed 0, N16)',name))
 from report_diagnostics import panels
 images.extend(panels(root,ts,figdir,METHODS,SIZES,LABELS,COLORS,STAGES,densities,records,finish))
 intro=(root/'report_template/intro.md').read_text()
 md=[intro,'\n## 현재 진행 상황\n',table(['단계','완료 실행 노드'],[(s,counts[s]) for s in counts]),
  f'\n**완료 비교 궤적 {sum(counts[s] for s in STAGES)}/448.** Prefix와 source는 비교 궤적 완료 수에 포함하지 않는다. 실패 기록 {len(failed)}개.\n',
  f'실행 소스 ID: `{manifest["code_id"]}`. [상세 프로토콜](PROTOCOL.md).\n',
  '## 사용한 Q와 환경\n','![Q 변경 일정](figures/00_q_schedules.png)\n','![학습 MDP reward](figures/01_mdp_reward.png)\n']
 if summaries:
  md+=['## 완료된 결과만의 요약\n','[전체 seed별 수치](summary.csv). 평균과 sample SD는 seed 단위로 계산한다. 부분 완료 평균은 최종 비교로 해석하지 않는다.\n']
  rows=[]
  for stage in STAGES:
   for n,m in SIZES:
    for method in METHODS:
     values=[s['binned_tv'] for s in summaries if (s['stage'],s['n'],s['method'])==(stage,n,method)]
     if values:rows.append([stage,f'{n}×{m}',LABELS[method],len(values),f'{np.mean(values):.4f}',f'{np.std(values,ddof=1):.4f}' if len(values)>1 else '—'])
  md+=[table(['종류','N×M','방법','완료 seeds','Histogram TV 평균','SD'],rows)]
 for label,im in images:md += [f'\n## {label}\n',f'![{label}](figures/{im})\n']
 md+=['\n## 해석과 남은 검증\n','현재 자동 보고서는 결과를 선별하지 않고 모은 자료다. OT 우위·GMM collapse의 결론은 density, teacher 오차, sigma/mean 분업과 seed별 실패를 함께 확인한 뒤 작성한다. 학습 중인 run의 최종 결과를 추정하거나 미리 채우지 않는다.\n']
 text='\n'.join(md);(out/'report.md').write_text(text)
 import shutil
 shutil.copy2(root/'PROTOCOL.md',out/'PROTOCOL.md')
 try:
  import markdown
  body=markdown.markdown(text,extensions=['tables','fenced_code'])
 except ImportError:body='<pre>'+html.escape(text)+'</pre>'
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>1D non-stationary Q</title><style>body{max-width:1280px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243248}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}pre{white-space:pre-wrap}</style>'+body)
 (out/'STATUS.json').write_text(json.dumps(dict(completed=counts,comparison_complete=sum(counts[s] for s in STAGES),failed=failed,code_id=manifest['code_id']),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(Path(a.root),Path(a.out))
