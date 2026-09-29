"""Post-hoc saved-trajectory audit; no training changes or new rollouts."""
import importlib.util,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('details',ROOT.parent/'antmaze_dacer_off_t1/report_completed.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
audit=mod.audit
snapshot=Path((ROOT/'latest-report-path.txt').read_text().strip())
load=lambda p:json.loads(p.read_text())
rows=[];arrays={}
for host in ['vast-heechan-180','vast-heechan-199']:
 for run in sorted((snapshot/host/'runs').glob('*')):
  cfg=load(run/'config.json'); task=cfg['task'];T=cfg['temperature']
  assert cfg['source_commit']=='5baa5b3463416cdfdcad465e8f862f3729802568'
  assert cfg['reward_profile']=='dense' and not cfg['noveld_enabled']
  hist=[]
  for raw in sorted(run.glob('evaluations/*/policy-natural/rollouts.npz')):
   d,m=mod.tools.read_mode(raw.parent,task)
   families=[mod.family(task,xy[:int(n)+1]) for xy,n in zip(d['xy'],d['lengths'])]
   m['corridors']=dict(Counter(families))
   m['mean_closest_goal']=float(np.mean([np.linalg.norm(xy[:int(n)+1,None,:]-np.array(audit.GOALS[task])[None,:,:],axis=-1).min() for xy,n in zip(d['xy'],d['lengths'])]))
   m['visited_bins']=len(set(tuple(b) for xy,n in zip(d['xy'],d['lengths']) for b in np.floor(xy[:int(n)+1]/.5).astype(int)))
   hist.append(m)
  if not hist:continue
  arrays[task,T]=(d,families)
  progress=load(run/'progress.json');result=load(run/'result.json') if (run/'result.json').exists() else {}
  modes={}
  for raw in sorted(raw.parent.parent.glob('*/rollouts.npz')):
   dm,mm=mod.tools.read_mode(raw.parent,task);modes[raw.parent.name]=mm
  rows.append(dict(task=task,T=T,id=run.name,host=host,completed=result.get('completed',False),progress=progress,budget=cfg['steps'],latest=hist[-1],history=hist,summary_history=[load(p) for p in sorted(run.glob('evaluations/*/policy-natural/summary.json'))],modes=modes))
rows.sort(key=lambda r:(r['task'],r['T']))
fig,axes=plt.subplots(3,3,figsize=(14,14))
fig.subplots_adjust(top=.91,bottom=.08,hspace=.37,wspace=.23)
for r in rows:
 ax=axes[['v1','v3','v4'].index(r['task']),[3,5,10].index(r['T'])]
 d,families=arrays[r['task'],r['T']];audit.decorate(ax,r['task'])
 for xy,n,g,fam in zip(d['xy'],d['lengths'],d['goals'],families):
  line=xy[:int(n)+1]; color=mod.COLORS[fam]
  ax.plot(line[:,0],line[:,1],c=color,alpha=.3,lw=.7)
  ax.scatter(*line[0],c=color,marker='^',s=8,alpha=.5)
  if not g:ax.scatter(*line[-1],c='#c73443',marker='x',s=12,alpha=.65)
 m=r['latest']
 ax.set_title(f'{r["task"].upper()} | T={r["T"]:g} | {m["step"]/1e6:.2f}M '+('FINAL' if r['completed'] else 'INTERIM')+f'\nSuccess {m["successes"]}/{m["episodes"]} | '+', '.join(f'{k}:{v}' for k,v in m['corridors'].items()),fontsize=10)
fig.suptitle('OptiQ temperature sweep | latest saved direct-policy trajectories\nDense reward, DACER OFF, NovelD OFF | one training seed (0)',fontsize=17,y=.975)
fig.text(.5,.022,'Random starts; random z + conditional sigma. Blue: upper/left; orange: lower/right; gray: no gate.\nAll failures included (red endpoints). 40 interim / 100 final episodes; training steps differ across panels.',ha='center',fontsize=11)
fig.savefig(snapshot/'latest_trajectories.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(12,11))
for i,task in enumerate(['v1','v3','v4']):
 for r in rows:
  if r['task']!=task:continue
  h=r['history'];xx=[m['step']/1e6 for m in h]
  sh=r['summary_history']; axes[i,0].plot([m['step']/1e6 for m in sh],[100*m['success_rate'] for m in sh],marker='.',label=f'T={r["T"]:g}')
  # fraction on less used side, failures included. Neither/both excluded from numerator.
  side=('upper','lower') if task!='v3' else ('left','right')
  axes[i,1].plot(xx,[100*min(m['corridors'].get(s,0) for s in side)/m['episodes'] for m in h],marker='.',label=f'T={r["T"]:g}')
 for j in [0,1]:
  axes[i,j].set(xlabel='Environment interactions (M)',ylabel='Episodes (%)',title=task.upper()+(' | Success' if j==0 else ' | Minority corridor (incl. failures)'),ylim=(-2,102 if j==0 else 52));axes[i,j].grid(alpha=.2);axes[i,j].legend()
fig.suptitle('Saved-policy performance and two-sided corridor persistence | one seed',fontsize=15)
fig.tight_layout(rect=(0,0,1,.965));fig.savefig(snapshot/'learning_and_routes.png',dpi=155);plt.close(fig)
output={'source':'5baa5b3463416cdfdcad465e8f862f3729802568','snapshot':str(snapshot),'raw_validation_passed':True,'runs':rows,'input_sha256':audit.INPUTS}
(snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
lines=['# OptiQ 온도 실험 경과','', 'Dense −nearest-distance, DACER OFF, NovelD OFF. seed0 한 개. 새로운 step penalty 보상은 적용하지 않았다.','', '|미로|T|학습 M / 예산 M|상태|평가 M|성공|실패 포함 통로|','|---|---:|---:|---|---:|---:|---|']
for r in rows:
 m=r['latest'];lines.append(f'|{r["task"]}|{r["T"]:g}|{r["progress"]["step"]/1e6:.3f}/{r["budget"]/1e6:.3f}|'+('완료' if r['completed'] else '진행')+f'|{m["step"]/1e6:.3f}|{m["successes"]}/{m["episodes"]}|{m["corridors"]}|')
lines+=['','중간 40회/최종100회 랜덤 시작 direct-policy 평가. 성공률과 실패 포함 통로 방문을 구분하며 동일 상태의 다중 경로를 자동으로 입증하지 않는다. 학습 step이 서로 달라 최신 결과만으로 온도별 최종 순위를 매기지 않는다.','',f'![궤적]({snapshot}/latest_trajectories.png)',f'![추이]({snapshot}/learning_and_routes.png)']
(snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
for r in rows:
 print(json.dumps({k:r[k] for k in ['task','T','completed','latest']},ensure_ascii=False))
