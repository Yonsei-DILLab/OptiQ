"""Render the completed, paired physical-state continuation diagnostic."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
R=ROOT/'v3_route_collapse_audit/controlled_continuation'
x=json.loads((R/'result.json').read_text());q=json.loads((R/'q_cross_comparison.json').read_text());replay=json.loads((R/'replay_audit.json').read_text())
proof=json.loads((R/'verification.json').read_text());assert proof['passed'] and replay['positive_rewards']==0
assert replay['checkpoint_sha256']==proof['checkpoint_proofs']['final']['sha256']
spec=importlib.util.spec_from_file_location('plot_tools',ROOT/'report_completed.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
colors=['#2463b7','#bd8217','#12a080','#9d57b7']
selected=x['selected_old_left_episodes'];color={f'old_episode{e}-left_gate':c for e,c in zip(selected,colors)}
labels=['2M','2.75M','final'];titles=['2.000M policy','2.750M policy','4.008M policy']
stats={}
fig,axes=plt.subplots(1,3,figsize=(12,7.4))
fig.subplots_adjust(top=.80,bottom=.19,left=.055,right=.99,wspace=.25)
for ax,label,title in zip(axes,labels,titles):
    mod.audit.decorate(ax,'v3');data=np.load(R/f'continuation_{label}.npz')
    rr=[r for r in x['continuations'][label] if r['source'].endswith('left_gate')]
    for row in rr:
        line=data['xy'][row['episode'],:row['length']+1];c=color[row['source']]
        ax.plot(line[:,0],line[:,1],c=c,alpha=.7,lw=1)
        ax.scatter(*line[0],marker='^',s=45,c=c,edgecolor='white',lw=.5,zorder=6)
        ax.scatter(*line[-1],marker='x',s=20,c=c,zorder=6)
    dist=np.array([np.linalg.norm(np.array(r['endpoint'])-[-12,12]) for r in rr])
    stats[label]=dict(episodes=len(rr),successes=sum(r['success'] for r in rr),mean_end_distance=float(dist.mean()),
        mean_closest_left_distance=float(np.mean([r['closest_left_goal'] for r in rr])),
        mean_return=float(np.mean([r['return_undiscounted'] for r in rr])),
        mean_discounted_return=float(np.mean([r['return_discounted'] for r in rr])))
    ax.set(xlim=(-15,-5.5),ylim=(1.5,14),title=f'{title}\nMean final goal distance: {dist.mean():.2f} m')
fig.suptitle('The later policy loses progress on the same left-corridor states\nFour identical physical states, three stochastic trials each, per checkpoint',y=.97,fontsize=15)
fig.text(.5,.035,'Position, posture, velocity, remaining episode time, and policy random seeds are paired.\nTriangles: starting states; crosses: endpoints; star: left goal. Colors match starting states across panels.\nFresh CPU inference; random z + conditional sigma; no training or external noise.',ha='center',fontsize=9)
fig.savefig(R/'paired_left_continuations.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,5.6))
fig.subplots_adjust(top=.8,bottom=.2,wspace=.3,left=.085,right=.98)
steps=np.array([2.000128,2.750208,4.008448]);values={}
for stage,c,title in [('left_gate','#2463b7','Left corridor entrance'),('late600','#bd8217','Farther along left corridor')]:
    ids=[i for i,s in enumerate(q['states']) if s['stage']==stage]
    means=[];mins=[]
    for lab in labels:
        row=q['comparison'][lab]['2M'];means.append(float(np.mean(np.array(row['mean_q'])[ids])));mins.append(float(np.mean(np.array(row['min_q'])[ids])))
    values[stage]=dict(mean_twin_q=means,min_twin_q=mins)
    axes[0].plot(steps,means,'o-',c=c,label=title)
axes[0].axhline(0,c='#b9233e',ls='--',lw=1)
axes[0].set(xlabel='Checkpoint interactions (M)',ylabel='Predicted Q (twin mean)',title='Same states AND same action candidates')
axes[0].legend(fontsize=8,loc='lower right');axes[0].grid(alpha=.18)
w=replay['rolling_replay_proxies'];axes[1].plot([z['step']/1e6 for z in w],[z['left_next_states']/10000 for z in w],color='#2463b7',marker='o')
axes[1].axvline((replay['last_training_left_xy_step']+1000000)/1e6,color='#b9233e',ls='--',lw=1)
axes[1].set(xlabel='Training interactions (M)',ylabel='Left-gate states in latest 1M samples (%)',title='Left-side data ages out of the buffer')
axes[1].grid(alpha=.18)
fig.suptitle('Value estimates drift outside the visited state distribution\nAll actual rewards are nonpositive; no entropy term in the TD target',y=.97,fontsize=14)
fig.text(.5,.025,'Q uses 256 actions per state from the frozen 2M actor, unchanged across all three critics.\nEach curve averages four fixed states. Positive Q is incompatible with the true discounted nonpositive-reward return.\nTiming and paired tests support forgetting; they do not identify the initial cause of preferring the right route.',ha='center',fontsize=9)
fig.savefig(R/'q_drift_and_replay.png',dpi=180);plt.close(fig)

report=dict(continuation_stats=stats,fixed_state_fixed_action_q=values,replay=replay,
    scope='Four left-moving episodes selected evenly from all 15 in fresh old-policy inference; 3 repeats per state per checkpoint.',
    limits='Paired continuation proves behavior change on tested states. Positive Q proves value error there. Replay loss is directly measured; attributing the first route choice causally still requires intervention.')
(R/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# v3: 동일 상태에서 경로 퇴행과 critic drift 검증','',
       '후기 정책이 왼쪽 경로 안에서도 이전보다 전진하지 못하는지 직접 확인했다. 학습 소스/체크포인트는 변경하지 않았다. 2M 정책을 CPU에서 새로 rollout한 40회 중 왼쪽15회에서 순서상 고르게4회를 선택하고, 해당 통로 진입 순간의 위치·자세·속도·시뮬레이터 시간·남은 episode 시간을 그대로 복원했다. 각 상태마다3회, 정책별12회다. 정책 RNG도 대응시켰다.', '',
       '|정책|평균 마지막 목표 거리|평균 최근접 목표 거리|평균 남은구간 return|성공|', '|---|---:|---:|---:|---:|']
for lab,row in stats.items():lines.append(f'|{lab}|{row["mean_end_distance"]:.2f}m|{row["mean_closest_left_distance"]:.2f}m|{row["mean_return"]:.1f}|{row["successes"]}/{row["episodes"]}|')
lines+=['', '원래 같은 왼쪽 진입 상태에서 2M 정책은 목표5m 부근까지 진행하지만, 최종 정책은8.1m 부근에서 종료한다. 이전 경로의 진행 능력이 실제로 퇴행했다. 재실행한 옛 왼쪽15회의 정체 구간은 torso 높이 중앙값약0.44~0.51m, upright cos약0.985~0.996로, 단순 넘어짐 설명은 이 자료와 맞지 않는다.',
        '', '## critic: 동일 상태와 동일 행동 입력', '',
        '2M actor에서 뽑은 상태당256개 행동을 고정하고, 세 checkpoint의 critic으로 같은 입력을 평가했다. 왼쪽 진입4상태 평균 twin-Q는 -804.4 → -681.4 → +97.2다. 두 critic 중 작은 값의 평균도 최종+62.7이다. 안쪽4상태에서는 -633.3 → -314.1 → +378.6이며 min-Q 평균도+311.5다.',
        'dense reward는 항상0이하이고, plain TD의 entropy 항은0이다. 따라서 이 입력에서 실제 할인 Q는0을 넘을 수 없다. 양수 예측은 가치 추정 오류다. 서로 다른 위치나 행동을 비교해 생긴 차이가 아니다.',
        '', '## 시간 순서와 replay', '',
        f'학습 중 마지막 왼쪽 gate(x<-8) 방문은 {replay["last_training_left_xy_step"]:,} step이다. 1M FIFO 버퍼에서 약2.613M에 마지막 왼쪽 표본도 밀려난다. 최종 실제 replay의1,000,000개 observation 중 x<-8은0개이며, x 최솟값은 {replay["minx"]:.4f}다. 최종 보상1M개 중 양수는0개다.',
        '', '|시점|직전1M개 수집 next-state 중 왼쪽 gate 표본|', '|---|---:|']
for w in replay['rolling_replay_proxies']:lines.append(f'|{w["step"]:,}|{w["left_next_states"]:,}|')
lines += ['', '## 해석', '',
          '확인된 문제는 방문 분포가 오른쪽으로 집중된 뒤, 왼쪽 구간의 경험이 제거되고, 그 구간의 actor 진행 능력 및 critic 가치 추정이 유지되지 않는 현상이다. Dense reward도 실제 방문해 replay에 들어온 transition에서만 TD 오차를 제공한다. 공유 신경망의 다른 상태 업데이트가 왼쪽 출력을 유지해 주지는 않는다.',
          '초기 오른쪽 선택의 유발 원인과 후속 망각은 구분한다. 이번 paired rollout은 퇴행을 직접 확인했고 동일 입력 critic 검사로 가치 오차를 입증했지만, +Q가 처음의 오른쪽 선택을 일으켰다고 결론내리지 않는다. replay 보존 개입 없이 망각의 모든 원인을 확정한 것도 아니다.',
          '기존의 경로별 평균 return 차이만으로 설명하는 것은 부족했다. 경로에 따라 비용 차이가 있더라도, 가던 구간의 능력 손실과 critic의 분포 밖 오류가 별도로 발생했음을 추가 확인했다.',
          '', f'![동일 상태 후속 rollout]({R}/paired_left_continuations.png)', '', f'![가치 추정과 replay]({R}/q_drift_and_replay.png)']
(R/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
(R/'report-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.suffix in ('.json','.md','.png') and p.name!='report-sha256.json'},indent=2)+'\n')
print(json.dumps(report,indent=2))
