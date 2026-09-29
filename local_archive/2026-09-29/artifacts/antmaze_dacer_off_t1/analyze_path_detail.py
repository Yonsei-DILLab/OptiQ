"""Inspect corridor choices and partial paths independently of goal success."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parent
SNAPSHOT=ROOT/'reports/20260924T081324Z'
OUT=SNAPSHOT/'path_detail'
OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('prior_route_tools',ROOT.parent/'antmaze_dacer_entropy_t1/analyze_rollouts.py')
tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
audit=tools.audit
source=json.loads((SNAPSHOT/'analysis.json').read_text())
COLORS={'upper':'#2563eb','lower':'#db8b16','left':'#2563eb','right':'#db8b16',
        'both':'#9d3fae','uncommitted':'#687480'}
LABELS={'v1':{'upper':'Upper corridor','lower':'Lower corridor','uncommitted':'No corridor'},
        'v2':{'left':'Left corridor','right':'Right corridor','both':'Both sides','uncommitted':'No corridor'},
        'v3':{'left':'Left via y~+4','right':'Right via y~-8','both':'Both sides','uncommitted':'No side gate'},
        'v4':{'upper':'Upper entry','lower':'Lower entry','uncommitted':'No entry'}}
GEOMETRY={'v1':'First crossing x=-4: y>2 upper / y<-2 lower.',
          'v2':'Visited x<-4 / x>4. Both is a separate category.',
          'v3':'Visited x<-8 / x>8. Both is a separate category. Crossing y is retained.',
          'v4':'First crossing x=-4: y>2 upper / y<-2 lower; outer gate x=-12 retained.'}


def family(task,line):
    if task in ('v1','v4'):
        v=audit.crossing(line,-4)
        return 'upper' if v is not None and v>2 else 'lower' if v is not None and v<-2 else 'uncommitted'
    threshold=4 if task=='v2' else 8
    left=bool((line[:,0]<-threshold).any());right=bool((line[:,0]>threshold).any())
    return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'


def stats(x):
    if not len(x):return None
    a=np.asarray(x)
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))


data={};summary={}
for run in source['runs']:
    task=run['task'];path=audit.REPO/run['modes']['policy-natural']['input']
    d,m=tools.read_mode(path.parent,task)
    rows=[];goals=np.asarray(m['goal_coordinates'])
    for i,(xy,n,goal) in enumerate(zip(d['xy'],d['lengths'],d['goals'])):
        line=xy[:n+1];fam=family(task,line)
        distances=np.linalg.norm(line[:,None,:]-goals[None,:,:],axis=-1)
        nearest=distances.min(axis=1)
        gates={}
        for x in (-14,-12,-10,-8,-4,4,8):
            v=audit.crossing(line,x,side=-1 if x<0 else 1)
            if v is not None:gates[str(x)]=float(v)
        both=False;returned=False
        if task in ('v1','v4'):
            up=bool(((line[:,0]<-4)&(line[:,1]>2)).any())
            down=bool(((line[:,0]<-4)&(line[:,1]<-2)).any())
            # For v4, use the initial doorway, not a later goal-column overshoot.
            if task=='v4':
                up=bool(((line[:,0]<-4)&(line[:,0]>-6)&(line[:,1]>2)).any())
                down=bool(((line[:,0]<-4)&(line[:,0]>-6)&(line[:,1]<-2)).any())
            both=up and down
            entered=np.flatnonzero(line[:,0]<-4)
            returned=bool(len(entered) and (line[entered[0]:,0]>-2).any())
        else:
            both=fam=='both';threshold=4 if task=='v2' else 8
            entered=np.flatnonzero(np.abs(line[:,0])>threshold)
            returned=bool(len(entered) and (np.abs(line[entered[0]:,0])<threshold/2).any())
        rows.append(dict(episode=i,family=fam,success=bool(goal),goal=int(goal),length=int(n),
            initial_xy=line[0].tolist(),endpoint_xy=line[-1].tolist(),xy_min=line.min(0).tolist(),xy_max=line.max(0).tolist(),
            closest_goal_m=float(nearest.min()),closest_each_goal_m=distances.min(0).tolist(),
            endpoint_nearest_goal_m=float(nearest[-1]),first_closest_step=int(nearest.argmin()),
            path_length_m=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum()),
            final100_bbox_diagonal_m=float(np.linalg.norm(np.ptp(line[-100:],axis=0))),
            crossed_both_doorways=both,returned_to_central_zone_after_gate=returned,
            first_crossing_y=gates))
    group={}
    for fam in sorted(set(r['family'] for r in rows)):
        rr=[r for r in rows if r['family']==fam];fail=[r for r in rr if not r['success']]
        group[fam]=dict(episodes=len(rr),episode_ids=[r['episode'] for r in rr],successes=sum(r['success'] for r in rr),
            failures=len(fail),start_positive_y=sum(r['initial_xy'][1]>0 for r in rr),start_negative_y=sum(r['initial_xy'][1]<0 for r in rr),
            failed_closest_goal_m=stats([r['closest_goal_m'] for r in fail]),
            failed_last100_bbox_diagonal_m=stats([r['final100_bbox_diagonal_m'] for r in fail]),
            failed_within_1m=[r['episode'] for r in fail if r['closest_goal_m']<=1],
            failed_within_2m=[r['episode'] for r in fail if r['closest_goal_m']<=2],
            failures_more_than_2m=[r['episode'] for r in fail if r['closest_goal_m']>2],
            endpoint_median_xy=np.median([r['endpoint_xy'] for r in rr],axis=0).tolist())
    summary[task]=dict(step=m['step'],episodes=len(rows),definition=GEOMETRY[task],groups=group,
        crossed_both_doorways=sum(r['crossed_both_doorways'] for r in rows),
        returned_to_central_zone_after_gate=sum(r['returned_to_central_zone_after_gate'] for r in rows),
        episodes_detail=rows)
    data[task]=d

# Consistency checks anchor the interpretation to every saved trajectory.
assert summary['v3']['groups']['left']['episodes']==15
assert summary['v3']['groups']['right']['episodes']==24
assert len(summary['v3']['groups']['right']['failed_within_1m'])==9
assert summary['v4']['groups']['upper']['episodes']==38
assert summary['v4']['groups']['uncommitted']['episodes']==2
assert all(sum(g['episodes'] for g in s['groups'].values())==s['episodes'] for s in summary.values())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11.5,10.8))
fig.subplots_adjust(left=.065,right=.985,top=.87,bottom=.085,hspace=.4,wspace=.25)
for ax,(task,entry) in zip(axes.flat,summary.items()):
    audit.decorate(ax,task);d=data[task]
    for r in entry['episodes_detail']:
        line=d['xy'][r['episode'],:r['length']+1];color=COLORS[r['family']]
        ax.plot(line[:,0],line[:,1],color=color,alpha=.4,lw=.8,zorder=2)
        ax.scatter(*line[0],marker='^',s=12,color=color,alpha=.55,zorder=4)
        ax.scatter(*line[-1],s=12,color=color,edgecolor='white',lw=.3,alpha=.8,zorder=5)
    label=' / '.join(f'{LABELS[task][k]} {v["episodes"]}' for k,v in entry['groups'].items())
    ax.set_title(f'{task.upper()} | {entry["step"]/1e6:.3f}M | ALL {entry["episodes"]} episodes\n'+label,fontsize=10)
    ax.legend(handles=[Line2D([0],[0],color=COLORS[k],lw=2,label=f'{LABELS[task][k]} ({v["episodes"]})') for k,v in entry['groups'].items()],loc='upper right' if task=='v3' else 'lower right',fontsize=8,framealpha=.92)
fig.suptitle('Corridor choices, including every unsuccessful trajectory\nColor = path family; goal completion does not determine the color',fontsize=16,y=.97)
fig.text(.5,.02,'Same checkpoints as the previous report. Triangles: random starts; circles: all endpoints; stars: goals.\nT=1, DACER OFF, dense reward, NovelD OFF; one training seed. No new rollout or learning.',ha='center',fontsize=10)
fig.savefig(OUT/'all_path_families.png',dpi=170);plt.close(fig)

# Show the later part of incomplete routes and the step-by-step distance trace.
fig,axes=plt.subplots(2,2,figsize=(12,9.5))
fig.subplots_adjust(top=.84,bottom=.14,left=.08,right=.98,hspace=.45,wspace=.27)
for row,task in enumerate(('v3','v4')):
    ax=axes[row,0];audit.decorate(ax,task);d=data[task];entry=summary[task]
    for r in entry['episodes_detail']:
        if r['success']:continue
        line=d['xy'][r['episode'],:r['length']+1];color=COLORS[r['family']]
        ax.plot(line[:,0],line[:,1],color=color,lw=.7,alpha=.20)
        ax.plot(line[-100:,0],line[-100:,1],color=color,lw=1.25,alpha=.85)
        ax.scatter(*line[-1],color=color,edgecolor='white',lw=.4,s=22,zorder=6)
    ax.set_title(f'{task.upper()} | incomplete paths: last 100 steps highlighted\nFaint line = earlier motion; filled dot = endpoint',fontsize=10)
    ax=axes[row,1]
    goals=np.asarray(audit.GOALS[task])
    for r in entry['episodes_detail']:
        line=d['xy'][r['episode'],:r['length']+1]
        distance=np.linalg.norm(line[:,None,:]-goals[None,:,:],axis=-1).min(1)
        ax.plot(np.arange(len(line)),distance,color=COLORS[r['family']],lw=.85,alpha=.4)
        ax.scatter(len(line)-1,distance[-1],color=COLORS[r['family']],s=8)
    ax.axhline(.5,color='#c5283d',ls='--',lw=1,label='Goal radius 0.5 m')
    ax.set(xlabel='Episode environment step',ylabel='Distance to nearest goal (m)',ylim=(0,22),title=f'{task.upper()} | every episode, until its actual end')
    ax.grid(alpha=.18)
    ax.legend(handles=[Line2D([0],[0],color=COLORS[k],lw=2,label=LABELS[task][k]) for k in entry['groups']]+[Line2D([0],[0],color='#c5283d',ls='--',label='Goal radius 0.5 m')],fontsize=8,loc='upper right')
fig.suptitle('Partial-route detail: where progress ends\nSpatial motion alone cannot identify a fall, posture problem, or value-learning cause',fontsize=15,y=.97)
fig.text(.5,.025,'v3 and v4: the same saved 2.000M policies, 40 random-start episodes each.\nLeft: all incomplete episodes, without selecting examples. Right: all episodes; no padding after termination.',ha='center',fontsize=10)
fig.savefig(OUT/'incomplete_path_detail.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
for ax,task in zip(axes,('v1','v3')):
    entry=summary[task]
    for fam,g in entry['groups'].items():
        rr=[r for r in entry['episodes_detail'] if r['family']==fam];xy=np.asarray([r['initial_xy'] for r in rr])
        ax.scatter(xy[:,0],xy[:,1],s=37,c=COLORS[fam],alpha=.85,edgecolors='white',lw=.5,label=f'{LABELS[task][fam]} ({len(rr)})')
    ax.axhline(0,color='#b8c0cb',lw=.7);ax.axvline(0,color='#b8c0cb',lw=.7)
    ax.set(xlim=(-2.1,2.1),ylim=(-2.1,2.1),aspect='equal',xlabel='Initial x (m)',ylabel='Initial y (m)',title=task.upper())
    ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Initial position versus observed corridor choice\nOne rollout per random initial state; this does not establish same-state stochastic diversity',fontsize=12)
fig.savefig(OUT/'start_position_choices.png',dpi=170);plt.close(fig)

out=dict(snapshot=str(SNAPSHOT),same_checkpoints_as_previous_report=True,
    analysis_scope='All saved direct-policy trajectories, independent of terminal success. Geometric corridor classification, not clustering action or pose.',
    validation_passed=True,input_sha256=audit.INPUTS,policies=summary,
    limits=['One trained seed and one rollout per random initial state.',
            'Small oscillations within one corridor are not counted as distinct routes.',
            'XY cannot diagnose body posture, a fall, or the cause of incomplete motion.',
            'This is policy evaluation, not cumulative behavior coverage during training.'])
(OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# 성공 여부와 분리한 경로 상세 분석','',
    '직전 보고와 동일한 checkpoint를 사용했다: v1 최종3.008M/100회, v2 2.250M/40회, v3·v4 2.000M/각40회. 원시 좌표를 재검증했으며 새 학습·rollout은 없다.',
    '','|환경|실패 포함 통로 선택|해석|','|---|---|---|',
    '|v1|위65, 아래34, 출발부 잔류1|두 우회로가 실제 사용된다. 시작 y>0은 위53/잔류1, y<0은 아래34/위12다.|',
    '|v2|오른쪽40, 왼쪽0|시작점 차이에 따른 흔들림은 있지만 모두 같은 중앙 통로로 오른쪽 목표에 간다.|',
    '|v3|왼쪽15, 오른쪽24, 중앙 상부1|성공 경로는 하나였으나 이동 경로는 두 갈래다. 왼쪽도 멀리 진행한다.|',
    '|v4|상단 입구38, 하단 입구0, 입구 미통과2|실패를 포함해도 상단 편중. 상단38개 중37개가 x=-12의 상단 외곽 통로를 통과한다.|',
    '', '## v3',
    '왼쪽15개는 x=-8을 y=3.78~5.11에서 통과하고 x≈-12 세로 통로로 올라간다. 종료점은 x=-13.1~-11.0, y=6.6~9.4 근처다. 목표(-12,12) 최근접 거리는 2.36~4.82m이며 15개 모두 도달하지 못한다. 마지막100step의 이동영역 bbox 대각선 중앙값은 약0.80m다. xy상 해당 위치 부근에 머무르며, 새로 다른 통로로 전환한 궤적은 관측되지 않았다.',
    '오른쪽24개는 x=8을 y=-9.10~-7.62에서 통과한다. 14개는 도달하며, 미도달10개 중9개도 목표1m 이내(0.5046~0.8971m)에 접근한다. 이9개는 이후 목표 옆 x≈12.6~13.0, y≈-13.0~-12.4에서 종료한다. 나머지1개는 (9.1,-7.6) 부근에서 끝나며 최근접 목표 거리는4.03m다. 별도1개는 중앙 상부(-1.4,6.1) 부근에 머문다.',
    '따라서 이 checkpoint의 v3은 성공 여부로만 보면 놓치는 양방향 진행이 있다. 다만 왼쪽 목표 도달과 동일 상태에서의 확률적 경로 선택은 확인되지 않았다.',
    '', '## v4',
    '38개가 x=-4를 y=3.21~4.84에서 통과한다. 이 중37개는 x=-12를 y=6.89~8.05로 건너는 상단 외곽 우회다. 중앙 가로 통로와 하단 대안 통로의 횡단은 없다. 1개는 x≈-10에서 진행을 끝낸다. 출발부 잔류2개를 제외한 미도달15개 중10개는 목표2m 이내까지 접근하고, 나머지5개는 상단 진행 중 일찍 멈춘다. 성공23/40이라는 숫자와 별개로, 관측된 이동은 한쪽 통로에 집중된다.',
    '', '## 해석 범위',
    '각 episode가 방문한 통로를 기준으로 분류했으며 성공을 분류 조건으로 쓰지 않았다. 같은 통로 안의 작은 흔들림을 여러 모드로 부풀리지 않았다. v3의 두 갈래는 랜덤 시작점에 대한 경로 다양성이고, 같은 초기상태에서 두 갈래가 나오는지와는 다른 질문이다. 좌표만으로 넘어짐·자세 문제·critic 원인을 단정하지 않는다.',
    '', f'![전체 경로]({OUT}/all_path_families.png)', '', f'![미완료 경로 상세]({OUT}/incomplete_path_detail.png)', '', f'![시작점과 선택 경로]({OUT}/start_position_choices.png)']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT),summary={t:{'groups':s['groups'],'both':s['crossed_both_doorways'],'returned':s['returned_to_central_zone_after_gate']} for t,s in summary.items()}),indent=2))
