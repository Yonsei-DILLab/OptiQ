from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
out=Path((ROOT/'latest-metrics-path.txt').read_text().strip());data=json.loads((out/'analysis.json').read_text())
runs={r['id']:r for h in data['hosts'] for r in h['runs']}
selected=[('v1-optiq-progress100_geodesic_no_bonus-T1-s0','V1: geodesic, no bonus',('upper','lower')),
('v3-optiq-progress100_euclidean-T1-s0','V3: Euclidean + bonus',('left','right')),
('v3-optiq-progress100_geodesic_no_bonus-T1-s0','V3: geodesic, no bonus',('left','right')),
('v4-optiq-progress100_geodesic_no_bonus-T1-s0','V4: geodesic, no bonus',('upper','lower'))]
fig,axs=plt.subplots(2,2,figsize=(10.5,7.3));fig.subplots_adjust(top=.84,bottom=.11,wspace=.24,hspace=.44)
for ax,(key,title,sides) in zip(axs.flat,selected):
 hist=runs[key]['history'];steps=[h['step']/1e6 for h in hist]
 for side,color in zip(sides,['#2266bb','#e18a24']):
  ax.plot(steps,[h['corridors'].get(side,0) for h in hist],'-o',label=side.capitalize()+' corridor',color=color,lw=2,ms=4)
 ax.plot(steps,[h['successes'] for h in hist],'--',label='Success, any corridor',color='#242b33',lw=1.6)
 ax.set(title=title,xlabel='Environment interactions (M)',ylabel='Episodes / 40',ylim=(-1,41));ax.grid(alpha=.18)
 ax.legend(fontsize=8,loc='best')
fig.suptitle('Saved policy corridor use and success over training\nOptiQ progress x100 | T=1 | DACER / NovelD OFF | seed 0',fontsize=14,y=.97)
fig.text(.5,.015,'Each point: 40 random-start direct-policy rollouts. Solid lines include successes and failures.\nDifferent policies are never pooled. No smoothing or training changes.',ha='center',fontsize=9)
fig.savefig(out/'corridor_progress.png',dpi=160);plt.close(fig)
lines=['# Progress100 경과 — 2026-09-24 23:05 KST 수집','',
'12개 실행/4개 대기/완료0/실패0. 모든실행PID생존. Native budget v1/v2 3M, v3 4M, v4 5M 유지.','',
'각행은최신저장40회 random-start direct-policy(random z+conditional sigma)평가다. 조건별학습step이다르며단일seed0이다.','',
'|미로|조건|평가 step|성공|통로 사용(실패 포함)|성공 통로|','|---|---|---:|---:|---|---|']
for r in sorted(runs.values(),key=lambda x:(x['task'],x['profile'])):
 h=r['latest']
 if h:lines.append(f"|{r['task']}|{r['profile']}|{h['step']}|{h['successes']}/40|{h['corridors']}|{h['successful_corridors']}|")
lines+=['', 'v3 Euclidean+B는1.25M에왼쪽30/오른쪽3으로양쪽성공했으나,1.5M/1.75M/2M평가에서는오른쪽통로방문0이다. 성공은36/40,36/40,34/40으로높게유지되므로성공률만보면이변화를놓친다.',
'', 'v1 geodesic bonus OFF는500k에위23/아래15성공,750k에위27/아래11성공으로두성공경로가유지된다. v4 geodesic은양쪽통로를쓰지만성공은아래쪽뿐이며, v2와v3 geodesic은오른쪽에집중한다.',
'', 'Remote saved raw NPZ를직접읽어유한좌표,랜덤초기상태,성공카운트와reward sum=100*(d0-dT)-T+B를검증하고각파일SHA256을보존했다. 큰NPZ를재전송하지않은원격읽기검증이며원자료는각raw_path에서유지된다. 새평가나학습변경없음.',
'', '![통로 추이](corridor_progress.png)']
(out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(out)
