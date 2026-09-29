"""Read-only reward/path audit. No rollout, optimizer, or server file writes."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'v3_route_collapse_audit'
OUT.mkdir(exist_ok=True)
REMOTE = r'''
import json, hashlib
from pathlib import Path
import numpy as np
from collections import Counter
r=Path('/home/heechan/optiq-experiments/antmaze-optiq-dense-dacer-off-T1-s0-20260924/runs/v3-optiq-dacer-off-T1-s0')
s=Path('/home/heechan/OptiQ-ops/sources/484f92e7d6d34c964d85b4493ff17c5a9ebcf32e')
success=json.loads((r/'training-successes.json').read_text())
xy=np.load(r/'training-xy.npy',mmap_mode='r')
windows=[]
for lo,hi in [(0,1000000),(1000000,1500000),(1500000,2000000),(2000000,2250000),(2250000,2500000),(2500000,2750000),(2750000,3000000),(3000000,len(xy))]:
 x=xy[lo:hi]
 windows.append(dict(start=lo,end=hi,goal_hits=dict(Counter(v['goal'] for v in success if lo<v['step']<=hi)),left_gate_state_fraction=float((x[:,0]<-8).mean()),right_gate_state_fraction=float((x[:,0]>8).mean())))
files={}
for rel in ['analysis_tools/experiments/20260920_truncated_mll/optiq_dime/algorithm.py','antmaze_experiments/envs.py','antmaze_experiments/run.py','antmaze_experiments/learners.py','antmaze/ddiffpg/env/d4rl/locomotion/goal_reaching_env.py','antmaze/ddiffpg/__init__.py']:
 p=s/rel;files[rel]=dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
hashes={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ['training-successes.json','training-xy.npy','learner/progress.csv','config.json']}
print(json.dumps(dict(run=str(r),source=str(s),goal_success_counts=dict(Counter(v['goal'] for v in success)),first_goal_success={str(g):next((v for v in success if v['goal']==g),None) for g in (1,2)},training_windows=windows,csv=(r/'learner/progress.csv').read_text(),config=json.loads((r/'config.json').read_text()),sources=files,input_sha256=hashes)))
'''
call = subprocess.run(['ssh','vast-heechan-180','/home/heechan/.venv-ddiffpg-native/bin/python','-'],
                      input=REMOTE,text=True,capture_output=True,check=True,timeout=50)
remote=json.loads(call.stdout)
(OUT/'remote-evidence.json').write_text(json.dumps(remote,indent=2)+'\n')
(OUT/'learner-progress.csv').write_text(remote['csv'])
for rel,entry in remote['sources'].items():
    p=OUT/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(entry['text'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']

spec=importlib.util.spec_from_file_location('routes',ROOT/'report_completed.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base=ROOT/'reports/20260924T085941Z/vast-heechan-180/runs/v3-optiq-dacer-off-T1-s0'
rollouts=[]
for raw in sorted((base/'evaluations').glob('*/policy-natural/rollouts.npz')):
    d,m=mod.tools.read_mode(raw.parent,'v3');rows=[]
    for i,n in enumerate(d['lengths']):
        line=d['xy'][i,:n+1]
        reward=-np.linalg.norm(line[1:,None,:]-np.asarray(mod.audit.GOALS['v3'])[None,:,:],axis=-1).min(-1)
        np.testing.assert_allclose(reward.sum(),d['returns'][i],rtol=2e-6,atol=.005)
        rows.append(dict(episode=i,route=mod.family('v3',line),length=int(n),success=bool(d['goals'][i]),
                         start_xy=line[0].tolist(),return_undiscounted=float(reward.sum()),
                         return_discounted_099=float(np.sum(reward*.99**np.arange(n))),
                         mean_step_reward=float(reward.mean())))
    groups={}
    for name in sorted(set(r['route'] for r in rows)):
        rr=[r for r in rows if r['route']==name]
        groups[name]=dict(episodes=len(rr),successes=sum(r['success'] for r in rr),
            **{k:float(np.mean([r[k] for r in rr])) for k in ['length','return_undiscounted','return_discounted_099','mean_step_reward']})
    rollouts.append(dict(step=m['step'],groups=groups,episodes=rows,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))

records=list(csv.DictReader(io.StringIO(remote['csv'])))
metrics=['train/temperature','train/actor_std_mean','train/actor_std_at_max_fraction','train/source_ess_absolute','train/max_source_weight','train/source_q_std','train/backup_entropy_term','train/actor_between_mean_variance']
log_groups=[]
for step in sorted(set(int(float(r['time/total_timesteps'])) for r in records)):
    rr=[r for r in records if int(float(r['time/total_timesteps']))==step]
    log_groups.append(dict(step=step,rows=len(rr),**{k:float(np.mean([float(r[k]) for r in rr])) for k in metrics}))
assert all(r['train/temperature']==1 and r['train/backup_entropy_term']==0 for r in log_groups)
result=dict(source=remote['source'],run=remote['run'],training_windows=remote['training_windows'],
            training_successes=remote['goal_success_counts'],first_success=remote['first_goal_success'],
            evaluations=rollouts,learner_logs=log_groups,config=remote['config'],input_sha256=remote['input_sha256'],
            limitations=['Grouped rollout returns use different sampled starting states; they are not a same-state causal Q comparison.',
                        'Discounted finite rollout sums omit value beyond time-limit truncation and must not be equated with critic Q.',
                        'Training XY gate occupancy is a fraction of transitions, not an episode route fraction.',
                        'No direction-conditioned critic prediction or controlled intervention is measured.',
                        'Single training seed.'])
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# v3 경로 집중 원인 점검','',
       '확인: frozen source에서 dense reward는 다음 위치와 가장 가까운 목표의 거리의 음수다. 원래 sparse goal bonus는 대체된다. 목표 반경0.5m에 도달하면 학습/평가 모두 terminal이며 TD bootstrap이0이 된다. 시간제한 종료는 bootstrap을 유지한다.', '',
       '|step|경로|평가수|성공|평균 길이|누적 보상|할인 누적 보상 γ=.99|', '|---:|---|---:|---:|---:|---:|---:|']
for row in rollouts:
    if row['step'] not in [1250048,1500160,2000128,2250240,2500096,2750208,4008448]:continue
    for route,g in row['groups'].items():
        lines.append(f'|{row["step"]}|{route}|{g["episodes"]}|{g["successes"]}|{g["length"]:.1f}|{g["return_undiscounted"]:.1f}|{g["return_discounted_099"]:.1f}|')
lines += ['', '학습 성공: 왼쪽 목표0, 오른쪽10830. 최초 오른쪽 성공은1,252,892 interactions. 1M~1.5M 구간 학습 transition의 왼쪽 gate 밖 점유율15.51%, 1.5M~2M 0.60%, 2M이후0%. 평가의 랜덤 시작점 분포와 학습의 원본 중앙 고정 시작 분포는 다르다.',
          '', 'T는1로 일정하며 σ 파라미터 평균은집중시기에도약0.36이다. 2.72M 로그에서 약97%의 좌표가 상한σ=exp(-1)에 해당한다. 작은sigma가 갑자기0이 된 현상으로 설명할 근거는 없다. 이 숫자는 minibatch 평균이며 모든 상태에서의 entropy를 뜻하지 않는다.',
          '', 'critic은 plain TD이며 미래 entropy 항은0이다. actor는 softmax(Q/T - beta*log proposal_density)로 만든 가중치를 Direct GMM NLL로 모사한다. 이 구조의 상태별 행동 분포 폭과 장기 경로 점유율은 서로 다르며, 왼쪽/오른쪽을 일정 비율로 유지하는 항은 없다.',
          '', '해석: 오른쪽의 더 낮은 거리 비용과 실제 진행/종료 이점, 이후 오른쪽으로 집중된 학습 경험이 경로 편중을 강화한 설명과 관측이 일치한다. 다만 방향별 동일 상태 Q 예측이나 개입 실험을 하지 않았으므로 critic 편향과 실제 경로 가치의 기여를 인과적으로 분리한 것은 아니다.',
          '', '위 그룹 평균은 시작 상태가 다른 평가를 묶은 값이므로 이 차이를 그대로 동일 상태의 Q 차이로 쓰지 않는다. 할인 누적 보상은 시간제한 이후의 가치를 포함하지 않는 유한 합이다. 원자료·소스·SHA256은 같은 폴더에 보관한다.']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT),training_successes=result['training_successes'],logs=[r for r in log_groups if r['step'] in (1280000,1920000,2400000,2720000,4000000)]),indent=2))
