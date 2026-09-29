"""Verified final-checkpoint report; historical seed 0 is explicitly identified."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
WORK=BASE.parents[1]/'tmp/pointmaze-multiseed-worktree'
sys.path.insert(0,str(WORK))
from maze_benchmarks.visualize_pointmaze import plot_map,plot_rollouts
from maze_benchmarks.evaluation_metrics import removal_from_raw
ARCH=BASE/'archive';OUT=BASE/'report';OUT.mkdir(exist_ok=True)
verified={p.stem:json.loads(p.read_text()) for p in (ARCH/'verified').glob('*.json')}
rows=[]
for name,v in sorted(verified.items()):
 d=dict(v);d.pop('sha256');d['path']=str(ARCH/'runs'/name);d['historical_seed0']=False;rows.append(d)

# Source changes only in evaluation/reporting and unrelated DIPO code are audited
# separately. Compare the full native learning configuration, excluding provenance.
ignored={'seed','output','output_root','run_name','source_commit','optiq_eval_mode'}
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ignored}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
compatibility=[]
for task in ['pm_medium','pm_hard']:
 for method in ['optiq','sac','sql','td3']:
  controls=[r for r in rows if r['task']==task and r['method']==method]
  if not controls:continue
  path=(BASE.parent/'pointmaze_optiq_t5_t10_1m_20260926/runs/pm_hard-optiq-t10-s0'
        if task=='pm_hard' and method=='optiq' else
        BASE.parent/'maze_deadline_1m_20260926/runs'/f'{task}-{method}-s0')
  c=json.loads((path/'config.json').read_text());current=json.loads((Path(controls[0]['path'])/'config.json').read_text())
  compatible=clean(c)==clean(current)
  compatibility.append(dict(task=task,method=method,path=str(path),compatible=compatible,source=c['source_commit'],comparison_source=current['source_commit']))
  if not compatible:continue
  p=json.loads((path/'progress.json').read_text());rec=p['latest_evaluation'];mode='mu_only' if method=='optiq' else 'policy';summary=rec[mode]
  assert p['status']=='complete' and p['steps']==1000192 and p['updates']==62000
  raw=path/'evaluations'/f"{p['steps']:09d}_{mode}.npz"
  with np.load(raw) as data:
   ids=data['goal_ids'];assert np.bincount(ids[ids>=0],minlength=len(summary['goals'])).tolist()==summary['goals']
  rows.append(dict(name=path.name,path=str(path),task=task,method=method,seed=0,temperature=c['temperature'],source_commit=c['source_commit'],steps=p['steps'],updates=p['updates'],mode=mode,primary=summary,record=rec,historical_seed0=True,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))

def panel(ax,r):
 if r is None:
  ax.text(.5,.5,'Final result not archived yet',ha='center',va='center');ax.axis('off');return
 metric=r['primary'];maze=r['task'][3:];path=Path(r['path'])/'evaluations'/f"{r['steps']:09d}_{r['mode']}.npz"
 plot_map(ax,maze,goal_counts=metric['goals'],outcome_colors=True)
 plot_rollouts(ax,path,max_trajectories=metric['episodes'],alpha=.42,outcome_colors=True,success_color='#c51b8a',failure_color='#e87924')
 ax.set_title(f"{maze.title()} | seed {r['seed']} | {metric['success']:.1%}\n{r['steps']:,} steps | {sum(x>0 for x in metric['goals'])}/{len(metric['goals'])} goals",fontsize=11)
 ax.set_xlabel(str(metric['goals']),fontsize=9);ax.set_ylabel('');ax.set_xticks([]);ax.set_yticks([])
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)

for method,seeds in [('dipo',[0,1,2]),('optiq',[0,1,2,3]),('sac',[0,1,2,3]),('sql',[0,1,2,3]),('td3',[0,1,2,3])]:
 fig,axes=plt.subplots(2,len(seeds),figsize=(4.3*len(seeds),9.0))
 for i,task in enumerate(['pm_medium','pm_hard']):
  for j,seed in enumerate(seeds):panel(axes[i,j],next((r for r in rows if r['task']==task and r['method']==method and r['seed']==seed),None))
 subtitle=('Fresh initial Gaussian; reverse diffusion noise OFF' if method=='dipo' else 'Fresh random z, mean only; Medium T=3 / Hard T=10' if method=='optiq' else 'Native generator outputs')
 fig.suptitle(method.upper()+' | final checkpoints | '+subtitle,fontsize=13)
 fig.text(.5,.014,'All 500 recorded rollouts per policy; magenta = success, orange = failure. Each panel is one learned policy.',ha='center',fontsize=10)
 fig.subplots_adjust(top=.90,bottom=.075,hspace=.28,wspace=.15)
 save(fig,method+'_final_seeds')

simple=next((r for r in rows if r['task']=='pm_simple' and r['method']=='dipo'),None)
best=[simple]+[max((r for r in rows if r['task']==task and r['method']=='dipo'),key=lambda r:r['primary']['success'],default=None) for task in ['pm_medium','pm_hard']]
fig,axes=plt.subplots(1,3,figsize=(13.2,5.4))
for ax,r in zip(axes,best):panel(ax,r)
fig.suptitle('DIPO | best final seed per maze (post-hoc selection, not a seed average)',fontsize=13)
fig.text(.5,.02,'All 500 rollouts shown. Fresh initial Gaussian, reverse diffusion noise OFF. All panels use the final ~1M checkpoint.',ha='center',fontsize=9)
fig.subplots_adjust(top=.80,bottom=.14,wspace=.12);save(fig,'dipo_best_final_simple_medium_hard')

aggregates=[]
for task in ['pm_medium','pm_hard']:
 for method in ['optiq','sac','sql','td3','dipo']:
  group=sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r['seed'])
  if not group:continue
  success=np.array([r['primary']['success'] for r in group]);coverage=np.array([sum(x>0 for x in r['primary']['goals']) for r in group])
  for r in group:
   r['removal_sr5_recomputed']=removal_from_raw(Path(r['path'])/'evaluations',r['record'],r['mode'])
   obstacle='obstacle_'+r['mode']
   if obstacle in r['record']:
    with np.load(Path(r['path'])/'evaluations'/f"{r['steps']:09d}_{obstacle}.npz") as raw:
     ids=raw['goal_ids'];r['obstacle_sr5_from_raw']=float(np.mean(np.any(ids.reshape(-1,5)>=0,axis=1)))
  aggregates.append(dict(task=task,method=method,seeds=[r['seed'] for r in group],complete_seed_set=len(group)==(3 if method=='dipo' else 4),success_mean=float(success.mean()),success_sample_sd=float(success.std(ddof=1)) if len(success)>1 else None,reachable_mean=float(coverage.mean()),reachable_sample_sd=float(coverage.std(ddof=1)) if len(coverage)>1 else None,success_per_seed=success.tolist(),coverage_per_seed=coverage.tolist()))

fig,axes=plt.subplots(2,2,figsize=(12,7.6),constrained_layout=True)
for i,task in enumerate(['pm_medium','pm_hard']):
 for r in rows:
  if r['task']!=task or r['method']!='dipo':continue
  hist=[json.loads(p.read_text()) for p in sorted((Path(r['path'])/'evaluations').glob('*_summary.json'))]
  steps=[h['step']/1e6 for h in hist];scores=[h['policy']['success'] for h in hist];modes=[h['policy']['reachable_goals'] for h in hist]
  axes[i,0].plot(steps,scores,'o-',label=f"seed {r['seed']}")
  axes[i,1].plot(steps,modes,'o-',label=f"seed {r['seed']}")
 axes[i,0].set(title=task[3:].title()+' success',ylim=(-.04,1.04),ylabel='Success rate')
 axes[i,1].set(title=task[3:].title()+' goals reached',ylim=(-.3,4.3 if i==0 else 8.3),ylabel='Goals reached')
 for ax in axes[i]:ax.set_xlabel('Environment transitions (millions)');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Corrected DIPO | observed checkpoint performance (200 episodes; final 500)')
save(fig,'dipo_learning_curves')

methods=['optiq','sac','sql','td3','dipo'];colors=['#2465b4','#dc7f25','#248955','#9975b4','#c64372']
fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
for i,task in enumerate(['pm_medium','pm_hard']):
 for j,(metric,title) in enumerate([('success','Success rate'),('goals','Goals reached'),('removal','Removal SR5'),('obstacle','Obstacle SR5')]):
  ax=axes[i,j]
  for k,method in enumerate(methods):
   group=[r for r in rows if r['task']==task and r['method']==method]
   if len(group)!=(3 if method=='dipo' else 4):continue
   vals=[]
   for r in group:
    if metric=='success':vals.append(r['primary']['success'])
    elif metric=='goals':vals.append(r['primary']['reachable_goals'])
    elif metric=='removal':vals.append(r['removal_sr5_recomputed'])
    elif 'obstacle_sr5_from_raw' in r:vals.append(r['obstacle_sr5_from_raw'])
   if vals:
    ax.bar(k,np.mean(vals),color=colors[k],alpha=.8)
    if len(vals)>1:ax.errorbar(k,np.mean(vals),yerr=np.std(vals,ddof=1),color='black',capsize=3)
    ax.scatter(np.full(len(vals),k)+np.linspace(-.08,.08,len(vals)),vals,color='black',s=15,zorder=3)
    ax.text(k,-.30 if metric!='goals' else -.6,f'n={len(vals)}',ha='center',fontsize=8)
  ax.set_title(task[3:].title()+' | '+title)
  ax.set_xticks(range(5),[m.upper() for m in methods],rotation=25);ax.grid(axis='y',alpha=.2)
  if metric!='goals':ax.set_ylim(-.35,1.35)
  else:ax.set_ylim(-.8,10 if i else 5.2)
fig.suptitle('Final ~1M | mean and sample SD across training seeds; dots = individual policies\nOptiQ Medium T3 / Hard T10, random-z mean only; DIPO UTD1/64 vs other methods UTD1/16',fontsize=12)
save(fig,'final_metrics_seed_comparison')

fig,axes=plt.subplots(2,4,figsize=(16,7.7),constrained_layout=True)
for i,task in enumerate(['pm_medium','pm_hard']):
 for k,method in enumerate(methods):
  group=[r for r in rows if r['task']==task and r['method']==method]
  if len(group)!=(3 if method=='dipo' else 4):continue
  histories=[]
  for r in group:
   folder=Path(r['path'])/'evaluations'
   hist=[json.loads(p.read_text()) for p in sorted(folder.glob('*_summary.json'))]
   histories.append((r,folder,hist))
  for j,metric in enumerate(['success','goals','removal','obstacle']):
   values=[];xs=None
   for r,folder,hist in histories:
    if metric=='obstacle' and 'obstacle_'+r['mode'] not in hist[-1]:continue
    vals=[]
    for rec in hist:
     m=rec[r['mode']]
     if metric=='success':val=m['success']
     elif metric=='goals':val=m['reachable_goals']
     elif metric=='removal':val=removal_from_raw(folder,rec,r['mode'])
     else:
      with np.load(folder/f"{rec['step']:09d}_obstacle_{r['mode']}.npz") as raw:val=float(np.mean(np.any(raw['goal_ids'].reshape(-1,5)>=0,axis=1)))
     vals.append(val)
    x=np.array([rec['step'] for rec in hist])/1e6
    if xs is None:xs=x
    assert np.array_equal(xs,x)
    values.append(vals)
   arr=np.asarray(values);mean=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1)
   axes[i,j].plot(xs,mean,label=f'{method.upper()} (n={len(values)})',color=colors[k],linewidth=2)
   axes[i,j].fill_between(xs,mean-sd,mean+sd,color=colors[k],alpha=.13)
 for j,title in enumerate(['Success rate','Goals reached','Removal SR5','Obstacle SR5']):
  ax=axes[i,j];ax.set(title=task[3:].title()+' | '+title,xlabel='Environment transitions (millions)')
  ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
fig.suptitle('Checkpoint learning curves | mean ± sample SD across seeds\nOptiQ random-z mean only; Medium T3 / Hard T10. DIPO has one quarter as many learner updates.',fontsize=12)
save(fig,'learning_curves_seed_comparison')

report=dict(generated=time.time(),reporting_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),posthoc_report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),archived_current=len(verified),historical_compatibility=compatibility,rows=rows,aggregates=aggregates)
(OUT/'results.json').write_text(json.dumps(report,indent=2))
lines=['# PointMaze 최종 평가 보고','',f'현재 승인 실험 36개 중 로컬 검증 보관: {len(verified)}개. 완료되지 않거나 보관 중인 작업은 최종 집계에서 제외한다.','', '| 환경 | 방법 | seeds | 성공률 평균 ± 표본SD | 목표 수 평균 ± 표본SD | 최종 시드 모두 포함 |','|---|---|---|---|---|---|']
for a in aggregates:
 sd=a['success_sample_sd'];gs=a['reachable_sample_sd']
 lines.append(f"| {a['task'][3:]} | {a['method']} | {a['seeds']} | {a['success_mean']*100:.2f} ± {(sd or 0)*100:.2f}% | {a['reachable_mean']:.2f} ± {gs or 0:.2f} | {a['complete_seed_set']} |")
lines+=['','OptiQ: Medium T3, Hard T10, random-z μ-only. DIPO: 공식 fresh initial Gaussian / reverse diffusion noise OFF. 다른 방법은 각 native generator. 각 패널과 목표 수는 한 정책의 반복 rollout으로 계산했다.','', 'DIPO는 seed0–2(3개), 다른 네 방법은 seed0–3(4개). 역사적 seed0는 전체 학습 설정을 대조하여 호환되는 결과만 포함하고 source commit을 별도로 기록했다. OptiQ Medium seed0의 장애물 평가는 σ 포함이라 μ-only 장애물 평가와 합산하지 않는다.','', 'DIPO 2048env/32updates, 15,520 learner updates와 다른 방법 256env/16updates, 62,000 learner updates는 약 1M 환경 전이 예산만 같으며 연산량·업데이트 수가 같지 않다. 수정 DIPO는 replay 영구 writeback 제거와 UTD 축소가 동시에 적용되어 단일 원인의 인과효과로 해석하지 않는다.','', 'Removal robustness는 원시 목표 ID로 SR5 조합식을 재계산한 지표이며 별도 재학습 결과가 아니다. 학습 경과별 최고점을 고른 이전 그림과 이 최종 checkpoint 그림을 구분한다.','', '## 수정 DIPO Way 최종 결과 (seed0)','', '| 환경 | 성공률 | 도달 목표 수 | 원시 목표별 횟수 |','|---|---|---|---|']
for r in sorted(rows,key=lambda x:x['task']):
 if r['method']=='dipo' and r['task'].endswith('way'):
  g=r['primary']['goals'];lines.append(f"| {r['task']} | {r['primary']['success']:.1%} | {sum(x>0 for x in g)}/{len(g)} | {g} |")
meow_paths=[BASE.parent/'maze_deadline_1m_20260926/runs/pm_simple-meow-s0']+list((BASE.parent/'pointmaze_simple_entropy_grid_1m/runs').glob('pm_simple-meow-*'))+[ARCH/'runs/pm_simple-meow-alpha05-s0']
meow=[]
for path in meow_paths:
 if not (path/'progress.json').exists():continue
 c=json.loads((path/'config.json').read_text());p=json.loads((path/'progress.json').read_text());rec=p['latest_evaluation'];metric=rec['policy']
 raw=path/'evaluations'/f"{rec['step']:09d}_policy.npz"
 raw_sha=None
 if raw.exists():
  with np.load(raw) as data:
   ids=data['goal_ids'];assert np.bincount(ids[ids>=0],minlength=4).tolist()==metric['goals']
  raw_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
 meow.append(dict(alpha=c['meow_alpha'],path=str(path),source=c['source_commit'],steps=rec['step'],success=metric['success'],goals=metric['goals'],raw_sha256=raw_sha,verification='raw-goal-counts' if raw_sha else 'historical-saved-summary-only'))
(OUT/'meow_simple_alpha_comparison.json').write_text(json.dumps(meow,indent=2))
lines+=['','## MEOW Simple 최종 α 비교 (각 seed0, 약 1M)','', '| α | 성공률 | 목표별 횟수 |','|---|---|---|']
for r in sorted(meow,key=lambda x:x['alpha']):lines.append(f"| {r['alpha']} | {r['success']:.1%} | {r['goals']} |")
lines+=['','α=.5는 성공률100%이지만 1/4 목표에 집중했다. 기존 α=.2 역시 성공률100%·1/4 목표다. α=1/3/10은 저장 요약에서 대부분 실패했으며 이 세 과거 대조군의 원시 NPZ는 이번 로컬 보관에 없어 요약 참고로만 표시한다. 새 α=.5는 원시 자료 및 checkpoint까지 SHA 검증했다. MFPO·MEOW 다중시드는 시작하지 않았다.']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(archived=len(verified),aggregates=aggregates,compatibility=compatibility),indent=2))
