from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
logs=json.loads((ROOT/'log-analysis.json').read_text())
returns=json.loads((ROOT/'route-returns.json').read_text())
rollouts=json.loads((ROOT/'saved-rollout-metrics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
for ax,run in zip(axes,rollouts['runs']):
 task=run['task']; hist=[r for r in run['history'] if r['step']<=1001000]
 pairs=[('left','Left','#3366cc'),('right','Right','#ee8a2e')] if task=='v3' else [('lower','Lower','#3366cc'),('upper','Upper','#ee8a2e')]
 for key,label,color in pairs:ax.plot([r['step']/1000 for r in hist],[r['corridors'].get(key,0)/r['episodes'] for r in hist],'-o',color=color,label=label,lw=2)
 ax.plot([r['step']/1000 for r in hist],[r['successes']/r['episodes'] for r in hist],'k--s',alpha=.7,label='Eval success')
 b=logs[task]['first_success_bracket'];lo=b['last_zero']['step']/1000;hi=b['first_positive']['step']/1000
 ax.axvspan(lo,hi,color='#ad55a5',alpha=.4);ax.axvline((lo+hi)/2,color='#ad55a5',linestyle=':',label='First training success')
 ax.set(title=f'{task}: route concentration vs first goal success',xlabel='Collected environment transitions (thousands)',xlim=(100,1020),ylim=(-.03,1.05));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='center right')
axes[0].set_ylabel('Fraction of 40 fixed-start direct-policy episodes')
fig.suptitle('Saved evaluations only: reward = 100 x Euclidean distance decrease; B=0, step cost=0',fontsize=11)
fig.tight_layout();fig.savefig(ROOT/'route_concentration_and_first_success.png',dpi=180);plt.close(fig)
lines=['# 저장 로그·궤적 진단','', '학습 소스: `f953d28456d3800860dddb9b9cb91b6bd520ae00`. 읽기 분석만 수행. 기존 학습·평가·설정을 변경하거나 새 평가를 실행하지 않았다. 4090은 사용하지 않았다.','', '## 가장 중요한 관측','', '**v4는 학습 중 goal 성공 이전에 이미 상단 경로로 집중됐다.** 따라서 “성공에 따른 terminal 이득이 발생한 순간에 경로가 사라졌다”는 설명은 v4에서 성립하지 않는다. 성공 이전의 연속적인 progress reward 또는 actor/critic 학습 편향은 여전히 가능하다.','']
for task in ('v3','v4'):
 d=logs[task];b=d['first_success_bracket'];lines += [f'## {task}','',f"첫 학습 성공은 total transitions ({b['last_zero']['step']:,}, {b['first_positive']['step']:,}] 사이. warmup 8,192를 제외한 global steps로는 ({b['last_zero']['global_steps']:,}, {b['first_positive']['global_steps']:,}] 사이다. 로그가 4,096 transition마다 기록되므로 정확한 1개 transition은 완료 시 training-successes.json에서 확인 가능하다.",'', '| 평가 total step | 경로 횟수 | 평가 성공 |','|---:|---|---:|']
 run=next(x for x in rollouts['runs'] if x['task']==task)
 for x in run['history']:
  if x['step']>1001000:break
  lines.append(f"| {x['step']:,} | {x['corridors']} | {x['successes']}/{x['episodes']} |")
 lines += ['', '250k에서 실제 trajectory로 재계산한 gamma=.99 return (평균±sample SD):','']
 for label,d in returns[task][0]['groups'].items():
  lines.append(f"- {label}: n={d['n']}, 할인 return={d['discounted_mean']:.2f} ± {d['discounted_sd']:.2f}, 무할인합계={d['undiscounted_mean']:.2f}, 최종 goal 거리={d['end_distance_mean']:.2f}m.")
 lines += ['', '| learner 로그 구간 | 평균 raw sigma | teacher ESS / 64 | 최대 teacher weight | 평균 source Q 표준편차 |','|---|---:|---:|---:|---:|']
 for w in logs[task]['windows'][:3]:
  m=w['means'];lines.append(f"| {w['logged_step_values']} | {m['train/actor_std_mean']:.5f} | {m['train/source_ess_absolute']:.2f} | {m['train/max_source_weight']:.3f} | {m['train/source_q_std']:.3f} |")
 lines += ['']
lines += ['## 해석 한계','', '- teacher ESS와 sigma는 replay minibatch를 평균낸 전역 로그다. 분기 상태에 국한된 teacher/actor 붕괴 여부를 판정하지 못한다. 그러나 전역 sigma가 0으로 줄거나 전역 teacher가 단일 후보로 집중된 흔적은 없다.', '- learner CSV는 160k transition 간격으로 각각 8개 연속 update를 기록한다. 구간 평균은 학습 전체의 시간평균이 아니다.', '- 250k 경로별 return은 사후에 경로로 분류한 rollout 표본이다. 동일 분기 상태에서 행동만 바꾸는 반사실적 비교가 아니며, critic Q 자체도 아니다.', '- v4 상단은 초기부터 더 깊이 진행했고 실현 return도 더 높았다. 이것은 지도의 본질적 상하 비대칭을 증명하지 않으며, 해당 시점 정책의 제어 능력 차이일 수 있다.', '- 학습 성공 카운트는 learner에 들어가는 training 환경의 성공 기록이고 evaluation의 성공은 이 카운터에 추가되지 않는다. 평가 환경이 별도로 생성됨을 run.py에서 확인했다.', '- 40회 중 한 경로가 0회라는 것은 그 확률이 엄밀히 0이라는 뜻이 아니다. 단일 seed 결과이며 확률적인 미관측 경로는 남아 있을 수 있다.','']
(ROOT/'REPORT_KO.md').write_text('\n'.join(lines))
provenance={'collected_utc':datetime.now(timezone.utc).isoformat(),'training_commit':'f953d28456d3800860dddb9b9cb91b6bd520ae00','remote_host':'vast-heechan-180','scope':'saved data read only; no new evaluation or training','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.name!='provenance.json'}}
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(ROOT/'REPORT_KO.md')
