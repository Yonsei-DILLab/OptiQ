"""Frozen-source trajectory report for the existing increasing-temperature schedule."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,hashlib,subprocess
from types import ModuleType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
A=ROOT.parent
W=A.parent/'tmp/reward-progress-worktree'
HELPER=A/'antmaze_geodesic_gamma999_250k/report_results.py'
spec=importlib.util.spec_from_file_location('geodesic_report',HELPER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SOURCE='acb0f2a85f9947de6500d6b1d7cae1d7944e3685'
CONTROL='8e9d7d3c2c79f797654ccfb21913d2c000b89f71'
rbytes=subprocess.check_output(['git','-C',str(W),'show',SOURCE+':antmaze_experiments/progress_reward.py'])
reward=ModuleType('increasing_report_frozen_reward');reward.__file__=str(W/'antmaze_experiments/progress_reward.py')
exec(compile(rbytes,reward.__file__,'exec'),reward.__dict__)
assert reward.UPSTREAM.read_bytes()==subprocess.check_output(['git','-C',str(W),'show',SOURCE+':antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py'])
base.reward=reward
OUT=ROOT/'report';TASKS=('v1','v4');MODES=('policy','native')
SCHEDULE=dict(enabled=True,final_temperature=3.,anneal_steps=1000000,decay='linear')
def read(p):return json.loads(p.read_text())
def collect():
 data={};proofs={};stops={}
 for name,root,source in (('fixed-T1',A/'antmaze_geodesic_gamma999_retention_1m',CONTROL),('linear-T1to3',ROOT,SOURCE)):
  host=root/'results/vast-heechan-180'
  if not (host/'manifest.json').exists():continue
  manifest=read(host/'manifest.json');assert manifest['source_commit']==source
  if (host/'screen-stop.json').exists():stops[name]=read(host/'screen-stop.json')
  for job in manifest['jobs']:
   task=job['task'];assert task in TASKS
   data[task,name]={mode:[] for mode in MODES}
   folder=host/'runs'/job['id']
   if not (folder/'config.json').exists():continue
   cfg=read(folder/'config.json');schedule=SCHEDULE if name=='linear-T1to3' else None
   assert cfg['source_commit']==source and cfg['seed']==0 and cfg['temperature']==1.
   assert cfg['native']['alg']['gamma']==.999 and cfg['discount']==.999
   assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
   assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
   assert cfg['temperature_schedule']==schedule
   assert cfg['native']['alg']['actor'].get('temperature_schedule')==schedule
   assert cfg['reward_profile']==reward.NO_COST_PROFILE and cfg['eval_starts']=='upstream'
   reset='natural' if task=='v1' else 'fixed'
   for mode in MODES:
    for path in sorted((folder/'evaluations').glob(f'*/{mode}-{reset}/summary.json')):
     if path.with_name('rollouts.npz').exists():
      item=base.evaluate(path,cfg,name,mode)
      item['row']['temperature']=1.+2*min(max((item['row']['step']-8192)/1e6,0),1) if schedule else 1.
      data[task,name][mode].append(item)
   if (folder/'result.json').exists():
    result=read(folder/'result.json')
    assert result['completed'] and result['source_commit']==source
    assert result['steps']==1008384 and result['updates']==31256
    assert result['checkpoint']['readback_verified'] and result['checkpoint']['environment_reward_verified']
    if schedule:assert result['final_temperature']==3. and result['temperature_schedule']==schedule
    proofs[task+'/'+name]=result['checkpoint']
 return data,proofs,stops

def main():
 OUT.mkdir(exist_ok=True);data,proofs,stops=collect()
 history=[e['row'] for modes in data.values() for es in modes.values() for e in es]
 latest=[es[-1]['row'] for modes in data.values() for es in modes.values() if es]
 matched=[]
 for mode in MODES:
  fig,axes=plt.subplots(2,2,figsize=(10,11),layout='constrained')
  for row,task in enumerate(TASKS):
   by={name:{e['row']['step']:e for e in data.get((task,name),{}).get(mode,[])} for name in ('fixed-T1','linear-T1to3')}
   common=by['fixed-T1'].keys() & by['linear-T1to3'].keys();step=max(common) if common else None
   for ax,name in zip(axes[row],by):
    item=by[name].get(step);base.draw(ax,task,item,f'{task} | {name}')
    if item:matched.append(item['row'])
  fig.suptitle(f'Original geodesic | gamma .999, H/d+.7 | {mode} | matched checkpoint per row\nv1 primary random starts; v4 original fixed full state; one training seed; no external noise')
  fig.savefig(OUT/f'matched_trajectories_{mode}.png',dpi=155);plt.close(fig)
  fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
  for row,task in enumerate(TASKS):
   for name in ('fixed-T1','linear-T1to3'):
    rows=[e['row'] for e in data.get((task,name),{}).get(mode,[])]
    if not rows:continue
    x=[r['step']/1000 for r in rows]
    for ax,key,scale in zip(axes[row],('success_rate','minority_success_rate','final_reward_distance_mean'),(100,100,1)):
     ax.plot(x,[scale*r[key] for r in rows],label=name,marker='.',ms=4)
   for ax,title in zip(axes[row],('Any-goal success (%)','Minority successful route (%)','Final goal distance (m)')):
    ax.set_title(task+' | '+title);ax.set_xlabel('Total transitions (k)');ax.grid(alpha=.2)
   axes[row,0].legend(fontsize=8)
  fig.suptitle(f'{mode} | all episodes/failures retained; v1 random starts do not prove same-state diversity')
  fig.savefig(OUT/f'learning_curves_{mode}.png',dpi=155);plt.close(fig)
 payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,control_source=CONTROL,
  goal_achieved=False,final_verified=proofs,screen_stops=stops,history=history,latest=latest,matched=matched,
  reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  reward_sha256=hashlib.sha256(rbytes).hexdigest(),helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
  caveats=['One training seed. Different settings are not independent seeds.',
           'v1 primary random-start diversity needs supplementary identical-origin verification.',
           'The fixed-T1 v4 control stopped early at626688; it has no final1M result.',
           'The linear-T1to3 v4 candidate stopped early after sustained minority-route loss; it has no final1M result.',
           'Native and direct-policy are kept separate; external behavior noise is absent.'])
 (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# 기존 선형 보간을 사용하는 온도 상승 실험','',
  '고정 T1과1→3 스케줄을 같은 보상·할인율에서 비교합니다. v1은 원래 랜덤 시작, v4는 원래 고정 full state입니다. v1의 양방향 성공만으로 동일 상태의 조건부 다중 경로를 주장하지 않습니다.','',
  '|환경|조건|mode|total step|학습 T|평가 수|진입|성공 통로|',
  '|---|---|---|---:|---:|---:|---|---|']
 for r in latest:lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['temperature']:.3f}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|")
 lines.extend(['','최종 checkpoint 검증: '+str(list(proofs)),
  '고정 T1 v4는626688에서 조기 중단한 대조군입니다. 중간 평가를 최종 결과로 바꾸지 않습니다.',
  'T1→3 v4도700k 이후 반대쪽 진입이40회 중0~1회로 줄고 성공이 없어서946176에서 중단했습니다. 최종1M 결과는 없습니다. v1의 완료 결과는 보존했습니다.',
  '![동일 step 직접 정책 비교](matched_trajectories_policy.png)'])
 (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(report=str(OUT),final_verified=list(proofs),rows=len(history))))
if __name__=='__main__':main()
