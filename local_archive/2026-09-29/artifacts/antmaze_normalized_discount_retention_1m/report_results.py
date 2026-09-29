"""Frozen-reward, same-budget provenance checks for v3 discount retention."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,importlib.util,json,subprocess
from types import ModuleType
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
A=ROOT.parent
W=A.parent/'tmp/reward-progress-worktree'
SOURCE='f5b3fcbfff3e61ecac4517facaa446393045c5a9'
CONTROL='db4ca0a446f5fe8df1dda02e261fa9154c66d9c9'
HELPER=A/'antmaze_startnorm_geodesic_250k/report_results.py'
spec=importlib.util.spec_from_file_location('verified_normalized_report',HELPER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=m.base
reward_path=W/'antmaze_experiments/progress_reward.py'
rbytes=subprocess.check_output(['git','-C',str(W),'show',SOURCE+':antmaze_experiments/progress_reward.py'])
assert rbytes==m.reward_bytes
reward=m.reward
base.reward=reward
OUT=ROOT/'report'
MODES=('policy','native')
CONDITIONS=('gamma999-short','gamma999-long','gamma99999-long')
def read(p):return json.loads(p.read_text())
def collect():
 runs={}
 for campaign,source,long in ((A/'antmaze_startnorm_geodesic_250k',CONTROL,False),(ROOT,SOURCE,True)):
  for host in (campaign/'results').glob('vast-heechan-*'):
   manifest=read(host/'manifest.json');assert manifest['source_commit']==source
   for job in manifest['jobs']:
    if not long and job['temperature']!=3.:continue
    name=('gamma999-long' if job['discount']==.999 else 'gamma99999-long') if long else 'gamma999-short'
    assert name in CONDITIONS and name not in runs
    r=dict(job=job,source=source,evaluations={mode:[] for mode in MODES},verified_final=False)
    runs[name]=r
    folder=host/'runs'/job['id']
    if not (folder/'config.json').exists():continue
    cfg=read(folder/'config.json')
    assert cfg['source_commit']==source and cfg['seed']==0 and cfg['task']=='v3'
    assert cfg['reward_profile']==reward.START_NORMALIZED_PROFILE
    assert cfg['reward_specification']==reward.specification('v3',reward.START_NORMALIZED_PROFILE)
    assert cfg['temperature']==3. and cfg['native']['alg']['gamma']==job['discount']
    assert cfg['discount']==job['discount'] and cfg['dacer_enabled'] and not cfg['noveld_enabled']
    assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
    assert cfg['eval_starts']=='upstream'
    for mode in MODES:
     for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
      if path.with_name('rollouts.npz').exists():r['evaluations'][mode].append(base.evaluate(path,cfg,name,mode))
    if (folder/'result.json').exists():
     proof=read(folder/'result.json')
     assert proof['completed'] and proof['source_commit']==source
     assert proof['steps']==job['steps'] and proof['updates']==(job['steps']-8192)//256*8
     assert proof['checkpoint']['environment_reward_verified'] and proof['checkpoint']['progress_replay_verified']
     r['verified_final']=True
 return runs

def main():
 OUT.mkdir(exist_ok=True)
 stop_path=ROOT/'results/vast-heechan-180/screen-stop.json'
 screen_stop=read(stop_path) if stop_path.exists() else None
 runs=collect();prefix=[]
 for mode in MODES:
  short={e['row']['step']:e for e in runs.get('gamma999-short',{}).get('evaluations',{}).get(mode,[])}
  long={e['row']['step']:e for e in runs.get('gamma999-long',{}).get('evaluations',{}).get(mode,[])}
  for step in sorted(short.keys() & long.keys()):
   p=np.load(short[step]['row']['raw_path']);q=np.load(long[step]['row']['raw_path'])
   for key in ('xy','returns','goals','lengths','initial_full_state'):np.testing.assert_array_equal(p[key],q[key])
   prefix.append(dict(mode=mode,step=step,arrays_exact=True))
 history=[e['row'] for r in runs.values() for es in r['evaluations'].values() for e in es]
 latest=[es[-1]['row'] for r in runs.values() for es in r['evaluations'].values() if es]
 for mode in MODES:
  fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
  for ax,name in zip(axes,CONDITIONS):
   es=runs.get(name,{}).get('evaluations',{}).get(mode,[])
   base.draw(ax,'v3',es[-1] if es else None,name)
  fig.suptitle(f'v3 | normalized geodesic progress, T3 | {mode} | latest saved checkpoint per condition\nOriginal identical full start, no external noise; one training seed; steps and failures shown')
  fig.savefig(OUT/f'latest_trajectories_{mode}.png',dpi=155);plt.close(fig)
  fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
  for name,r in runs.items():
   rows=[e['row'] for e in r['evaluations'][mode]]
   if not rows:continue
   x=[r['step']/1000 for r in rows]
   for ax,key,scale in zip(axes,('success_rate','minority_success_rate','closest_euclidean_goal_distance_mean'),(100,100,1)):
    ax.plot(x,[scale*r[key] for r in rows],marker='.',label=name)
  for ax,title in zip(axes,('Any-goal success (%)','Minority successful route (%)','Closest goal distance (m)')):
   ax.set_title(title);ax.set_xlabel('Total transitions (k)');ax.grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle(f'{mode} | all episodes including failures; matched curves, not independent seeds')
  fig.savefig(OUT/f'learning_curves_{mode}.png',dpi=155);plt.close(fig)
 payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,control_source=CONTROL,
  screen_stop=screen_stop,planned_long_budget_completed=False,
  algorithm_goal_achieved=False,final_verified=[k for k,r in runs.items() if r['verified_final']],
  exact_shared_prefix_checks=prefix,latest=latest,history=history,
  reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),reward_sha256=hashlib.sha256(rbytes).hexdigest())
 (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# v3 보정 보상: 할인율·장기 유지 비교','',
  '보상·T3·모델은 같고 .99999 조건에서 할인율만 바뀝니다. 같은 seed0의 짧은/긴 실험은 독립 seed가 아닙니다. 원래 고정 full state에서 평가하며 실패를 분모에 포함합니다. 성공한 양쪽 경로와 이후 유지를 확인하기 전에는 목표 달성으로 보고하지 않습니다.','',
  '|조건|mode|total step|평가 수|통로 진입|성공 통로|','|---|---|---:|---:|---|---|']
 for r in latest:lines.append(f"|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|")
 lines.extend(['','최종 전체 checkpoint 검증: '+str(payload['final_verified']),
  '같은 설정의 저장 궤적 prefix 일치 검사: '+str(len(prefix)),
  '최신 그림은 조건별 step을 표시합니다. 초기/중간40회를 최종100회로 부르지 않습니다.'])
 if screen_stop:
  lines.extend(['','두 장기 실험은 반대쪽 경로 상실이 반복되어 조기 중단했습니다. '
   '계획한 1M 완료나 최종100회 평가 결과로 해석하지 않습니다. 로그·중간 정책·중단 근거를 보존했습니다.',
   '중단 직전 기록 step: '+str({r['id']:r['last_logged_step'] for r in screen_stop['evidence']})])
 (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(report=str(OUT),final_verified=payload['final_verified'],shared_prefix_checks=len(prefix),rows=len(history))))
if __name__=='__main__':main()
