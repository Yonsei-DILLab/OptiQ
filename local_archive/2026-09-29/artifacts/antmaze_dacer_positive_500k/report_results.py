"""Read-only early/final route report from verified raw evaluation archives."""
from collections import Counter
import csv, hashlib, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'tmp/reward-progress-worktree'
sys.path.insert(0,str(SOURCE))
from antmaze_experiments.progress_reward import maze_geometry
from antmaze_experiments.critic_diagnostics import route_label
TRAINING_SHA='2564b59faa0d319eece496b93eff0f19359efc37'
TARGETS=(.1,.5,.7,.9)
TARGET_COLORS={.1:'#2779b7',.5:'#279765',.7:'#ee8a2b',.9:'#9861ad'}
TASKS=('v1','v2','v3','v4')
COLORS={'upper':'#2678b6','lower':'#ed8b32','left':'#2678b6','right':'#ed8b32','uncommitted':'#929aa4'}
OUT=ROOT/'report';OUT.mkdir(exist_ok=True)

def load(p):return json.loads(p.read_text())

def first_route(task,xy):
    if task in ('v1','v4'):return route_label(task,xy)[0]
    gate=4 if task=='v2' else 8
    passed=np.flatnonzero(np.abs(xy[:,0])>gate)
    if not len(passed):return 'uncommitted'
    return 'left' if xy[passed[0],0]<0 else 'right'

runs={};records=[];route_rows=[];verification=[];completion_proofs=[]
for host in sorted((ROOT/'results').glob('vast-heechan-*')):
    manifest=load(host/'manifest.json')
    assert manifest['source_commit']==TRAINING_SHA
    for job in manifest['jobs']:
        folder=host/'runs'/job['id'];path=folder/'config.json'
        if not path.exists():continue
        cfg=load(path);task=cfg['task'];target=cfg['dacer_target_entropy_per_dim']
        assert cfg['source_commit']==TRAINING_SHA and cfg['dacer_interval_updates']==500
        assert cfg['reward_specification']['formula']=='100*(d(current)-d(next))'
        if (folder/'result.json').exists():
            proof=load(folder/'result.json')
            assert proof['completed'] and proof['source_commit']==TRAINING_SHA
            assert proof['steps']==508416 and proof['updates']==15632
            assert proof['checkpoint']['readback_verified'] and proof['checkpoint']['environment_reward_verified']
            completion_proofs.append(dict(id=job['id'],host=host.name,steps=proof['steps'],updates=proof['updates'],
                                         checkpoint_sha256=proof['checkpoint']['sha256'],verified=True))
        primary='policy-natural' if task=='v1' else 'policy-fixed'
        evaluations=[]
        for path in sorted((folder/'evaluations').glob('*/'+primary+'/summary.json')):
            summary=load(path);raw=path.with_name('rollouts.npz')
            if not raw.exists():continue
            with np.load(raw,allow_pickle=False) as z:
                xy=z['xy'].copy();lengths=z['lengths'].copy();goals=z['goals'].copy()
                starts=z['initial_full_state'].copy();returns=z['returns'].copy()
                assert str(z['mode'])=='policy' and int(z['env_steps'])==summary['step']
            n=len(xy);assert n==summary['episodes'] and len(goals)==n
            assert np.isclose((goals>0).mean(),summary['success_rate'])
            assert np.isclose(returns.mean(),summary['mean_return'])
            if task!='v1':
                assert summary['original_origin_fixed'] and summary['identical_initial_full_state']
                np.testing.assert_array_equal(starts,np.repeat(starts[:1],n,axis=0))
                assert np.all(starts[:,:2]==0)
            else:
                assert not summary['fixed'] and not summary['identical_initial_full_state']
            points=[p[:int(k)+1] for p,k in zip(xy,lengths)]
            assert all(np.isfinite(p).all() for p in points)
            _,goal_xy,_=maze_geometry(task)
            d0=np.linalg.norm(np.asarray([p[0] for p in points])[:,None]-goal_xy,axis=-1).min(-1)
            dend=np.linalg.norm(np.asarray([p[-1] for p in points])[:,None]-goal_xy,axis=-1).min(-1)
            np.testing.assert_allclose(returns,100*(d0-dend),atol=.002,rtol=2e-5)
            for p,g in zip(points,goals):
                if g>0:assert np.linalg.norm(p[-1]-goal_xy[int(g)-1])<.501
            labels=[first_route(task,p) for p in points]
            entry=dict(Counter(labels));success=dict(Counter(l for l,g in zip(labels,goals) if g>0))
            goal_routes=dict(Counter(f'{l}/G{g}' for l,g in zip(labels,goals) if g>0))
            sides=('upper','lower') if task in ('v1','v4') else ('left','right')
            minority=min(entry.get(s,0) for s in sides)
            minority_success=min(success.get(s,0) for s in sides)
            record=dict(host=host.name,id=job['id'],task=task,target=target,
                step=summary['step'],global_step=summary['step']-8192,episodes=n,
                success_rate=float((goals>0).mean()),mean_return=float(returns.mean()),
                route_counts=entry,successful_route_counts=success,successful_goal_routes=goal_routes,
                goal_counts=summary['goal_counts'],minority_entry_rate=minority/n,
                minority_success_rate=minority_success/n,
                both_successful_routes_observed=minority_success>0,
                exploratory_screen_both_routes_ge10pct=minority_success>=max(4,.1*n),
                fixed_original_start=task!='v1')
            records.append(record)
            evaluations.append(dict(record=record,xy=points,goals=goals,labels=labels))
            verification.append(dict(path=str(raw.relative_to(ROOT)),sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                                     episodes=n,success_and_reward_verified=True,starts_verified=True))
        history=folder/'dacer_regulator_history.jsonl'
        entropy=[json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        runs[task,target]=dict(folder=folder,cfg=cfg,evaluations=evaluations,entropy=entropy)

latest=[r['evaluations'][-1]['record'] for r in runs.values() if r['evaluations']]
final_grid=len(latest)==16 and len(completion_proofs)==16 and all(r['step']==508416 and r['episodes']==100 for r in latest)
result=dict(training_source=TRAINING_SHA,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    evaluated_policies=len(latest),registered_policies=16,latest=latest,history=records,
    final_grid_complete=final_grid,completion_proofs=completion_proofs,
    criteria={'route_choice':'first gate passage, not union of all visited corridors',
              'success':'upstream success at original goal radius; verified against endpoint',
              'screen':'both routes each account for >=10% of all evaluation episodes, minimum4; screening only',
              'retention':'check consecutive later checkpoints and separate seeds; isolated crossing is insufficient'},
    limitations=['one training seed per condition','40 episodes at intermediate,100 final',
                 'v1 random starts: route differences alone do not prove same-state multimodality',
                 'training behavior noise is excluded from evaluation','no pooling separate policies'],
    raw_verification=verification)
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
fields=['task','target','step','global_step','episodes','success_rate','mean_return','minority_entry_rate','minority_success_rate','both_successful_routes_observed','route_counts','successful_route_counts','successful_goal_routes']
with (OUT/'history.csv').open('w') as f:
    writer=csv.DictWriter(f,fields);writer.writeheader()
    for row in records:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],dict) else row[k] for k in fields})
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(4,4,figsize=(16,15),layout='constrained')
for i,task in enumerate(TASKS):
    walls,goals,bounds=maze_geometry(task)
    for j,target in enumerate(TARGETS):
        ax=axes[i,j];run=runs.get((task,target));es=run['evaluations'] if run else []
        for x0,y0,x1,y1 in walls:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e3e6e9',edgecolor='#bcc2c8',lw=.4))
        if es:
            e=es[-1];r=e['record']
            order=np.argsort(e['goals']>0)
            for index in order:
                p=e['xy'][index];g=e['goals'][index];label=e['labels'][index]
                ax.plot(p[:,0],p[:,1],color=COLORS[label],alpha=.4 if g>0 else .17,lw=1 if g>0 else .6)
                ax.scatter(*p[-1],s=6,color=COLORS[label],alpha=.4)
            short={'left':'L','right':'R','upper':'U','lower':'D','uncommitted':'none'}
            counts=' '.join(f'{short[k]}:{v}' for k,v in r['route_counts'].items())
            sc=' '.join(f'{short[k]}:{v}' for k,v in r['successful_route_counts'].items()) or 'none'
            ax.set_title(f'{task} | H/d={target:g} | {r["step"]/1000:.0f}k\nentry {counts}\nsuccess {round(r["success_rate"]*r["episodes"])}/{r["episodes"]}: {sc}')
            ax.scatter(*np.array([p[0] for p in e['xy']]).T,s=6,c='black',alpha=.3,zorder=5)
        else:ax.set_title(f'{task} | H/d={target:g}\nNo saved policy evaluation yet')
        ax.scatter(*goals.T,marker='*',s=100,c='#38a35f',edgecolor='white',lw=.5,zorder=6)
        ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.suptitle('OptiQ DACER target sweep | latest saved direct-policy rollouts, seed0\nRandom z + conditional sigma; NO external DACER noise. v1 random starts; v2-v4 identical original full state.\nPanels can have different checkpoint steps. Pale trajectories include failures.',fontsize=14)
fig.savefig(OUT/'latest_trajectories.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(16,9),layout='constrained')
for col,task in enumerate(TASKS):
    for target in TARGETS:
        run=runs.get((task,target));es=run['evaluations'] if run else []
        if not es:continue
        x=[e['record']['step']/1000 for e in es]
        for row,key in enumerate(('success_rate','minority_entry_rate','minority_success_rate')):
            axes[row,col].plot(x,[e['record'][key] for e in es],marker='o',ms=3,color=TARGET_COLORS[target],label=f'H/d={target:g}')
    for row,title in enumerate(('Goal success fraction','Minority corridor entry / all episodes','Minority successful corridor / all episodes')):
        axes[row,col].set_title(task+' | '+title,fontsize=9);axes[row,col].set_ylim(-.02,1.02 if row==0 else .52)
        axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Total transitions (k)')
axes[0,0].legend(handles=[Line2D([0],[0],color=TARGET_COLORS[t],label=f'H/d={t:g}') for t in TARGETS],fontsize=8)
fig.suptitle('Goal-reaching and route retention are evaluated separately | one trained policy per line')
fig.savefig(OUT/'route_retention.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(16,9),layout='constrained')
for col,task in enumerate(TASKS):
    for target in TARGETS:
        run=runs.get((task,target));es=run['entropy'] if run else []
        if not es:continue
        x=[e['env_steps']/1000 for e in es]
        for row,key in enumerate(('entropy_proxy','noise_std','entropy_probe_clip_fraction')):
            y=[e[key]/8 if row==0 else e[key] for e in es]
            line=axes[row,col].plot(x,y,color=TARGET_COLORS[target],label=f'H/d={target:g}')[0]
            if row==0:axes[row,col].axhline(target,ls=':',lw=.8,c=line.get_color(),alpha=.5)
    for row,title in enumerate(('Measured joint-entropy proxy / d','External noise std (after update)','Clipped action fraction in entropy probe')):
        axes[row,col].set_title(task+' | '+title,fontsize=9);axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Total transitions (k)')
axes[0,0].legend(handles=[Line2D([0],[0],color=TARGET_COLORS[t],label=f'H/d={t:g}') for t in TARGETS],fontsize=8)
fig.suptitle('DACER control response | target is not exact action entropy or trajectory entropy')
fig.savefig(OUT/'entropy_response.png',dpi=160);plt.close(fig)
lines=['# DACER 양수 entropy '+('500k 전체 결과' if final_grid else '실험 중간 분석'),'',
('16개 모두 508416 total transitions(500224 post-warmup)와 최종 100회 평가·체크포인트 검증을 완료했습니다. ' if final_grid else '현재 수집 체크포인트는 패널별로 다를 수 있습니다. ')+
'지속적인 다중 성공 경로는 아직 입증되지 않았습니다. 각 패널은 서로 다른 단일 seed0 정책입니다.','',
'| 환경 | H/차원 | total step | 평가수 | 첫 통로 진입 | 성공 통로 | 성공률 |',
'|---|---:|---:|---:|---|---|---:|']
for item in sorted(latest,key=lambda r:(r['task'],r['target'])):
    lines.append(f"| {item['task']} | +{item['target']} | {item['step']} | {item['episodes']} | {item['route_counts']} | {item['successful_route_counts']} | {100*item['success_rate']:.1f}% |")
lines+=['','통로 진입과 goal 도달은 다릅니다. Random latent+conditional sigma 직접정책을 사용하고 외부 DACER 잡음은 평가에서 제외합니다. v1은 원래 랜덤 시작, v2-v4는 원래 고정 full state입니다.','',
        '![최근 궤적](latest_trajectories.png)','![경로 유지와 성공](route_retention.png)','![조절 반응](entropy_response.png)','',
        'Reward100*(d_current-d_next), bonus0·step penalty0·NovelD OFF, T1. 원시 NPZ episode/성공/goal endpoint/누적 reward를 검증했고 각 SHA256과 보고 코드 해시는 results.json에 보관했습니다. 학습 source2564b59는 그대로입니다.','',
        '후속 gamma.999/T3 및 조합의250k screen8개는 학습 sourceeee04de, controller7278a0e이며 핵심 알고리즘7파일은2564b59와 byte 단위 동일합니다. 완료된 v3 gamma.999/T1은 최종100회에서 left54/right33, success0이었고 mu-only에서도 left54/right20, success0입니다. 이 조건만 같은 seed0로 새로1M 학습하는 후속을 source d266263으로 등록했습니다. 이는 독립 seed 또는 checkpoint 재개가 아닙니다. 목표를 달성했다고 판정하지 않았습니다.']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'report':str(OUT),'policies':len(latest),'evaluations':len(records),'latest':latest},indent=2))
