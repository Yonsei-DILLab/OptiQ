"""Analyze verified v3 OptiQ archives without changing any training run."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from antmaze.multimodal.analysis import summarize, route_label

OUT = HERE / 'analysis' / 'v3-optiq-1m'
OUT.mkdir(parents=True, exist_ok=True)
ROOTS = {0.01: REPO / 'artifacts/antmaze_dense_noveld_1m/runs/v3-optiq-s0',
         0.1: HERE / 'runs/v3-optiq-s0'}
COLORS = {0.01: '#526fa3', 0.1: '#07877c'}
read = lambda p: json.loads(p.read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

def maze(ax, config):
    for x, y in config['environment']['walls']:
        ax.add_patch(Rectangle((x-2,y-2),4,4,facecolor='#414b59',edgecolor='#fafbfc',lw=.3,zorder=2))
    for i,(x,y) in enumerate(config['environment']['goals'],1):
        color = '#07877c' if i==1 else '#cd7d27'
        ax.add_patch(Circle((x,y),.5,color=color,zorder=6))
        ax.text(x+(1.2 if i==1 else -1.2),y+.4,f'G{i}',color=color,
                ha='left' if i==1 else 'right',weight='bold',fontsize=11,zorder=7,
                bbox={'fc':'white','ec':'none','alpha':.85,'pad':1})
    ax.scatter(0,0,marker='*',s=100,color='#182637',edgecolor='white',lw=.6,zorder=6)
    ax.set(xlim=(-18,18),ylim=(-18,18),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    ax.set_xticks([-12,-8,0,8,12]);ax.set_yticks([-12,-4,0,4,12])
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():s.set_visible(False)

def occupancy(xy):
    ij=np.floor((xy.astype(np.float64)+18)/.5).astype(int)
    ok=(ij>=0).all(1)&(ij<72).all(1)
    assert ok.all()
    ids=ij[:,0]*72+ij[:,1]
    return np.bincount(ids,minlength=72*72).reshape(72,72),ids

def verify_coverage_rounding(xy, stored, rebinned):
    # Coverage used float64 simulator positions; replay stores float32 observations.
    # Verify exact consistency allowing ONLY the half-ULP quantization intervals.
    q=xy.astype(np.float64)
    low=(q+np.nextafter(xy,np.float32(-np.inf)).astype(np.float64))/2
    high=(q+np.nextafter(xy,np.float32(np.inf)).astype(np.float64))/2
    a=np.floor((low+18)/.5).astype(int);b=np.floor((high+18)/.5).astype(int)
    ambiguous=np.any(a!=b,axis=1)
    fixed,_=occupancy(xy[~ambiguous]);need=(stored-fixed).ravel().copy()
    assert (need>=0).all() and need.sum()==int(ambiguous.sum())
    candidates=[]
    for lo,hi in zip(a[ambiguous],b[ambiguous]):
        candidates.append([i*72+j for i in range(lo[0],hi[0]+1) for j in range(lo[1],hi[1]+1)])
    def match(i):
        if i==len(candidates):return not need.any()
        for cell in candidates[i]:
            if need[cell]>0:
                need[cell]-=1
                if match(i+1):return True
                need[cell]+=1
        return False
    assert match(0),'Stored coverage cannot be explained by float32 quantization'
    return dict(float32_rounding_compatible=True,ambiguous_samples=int(ambiguous.sum()),
                rebinned_count_l1_difference=int(abs(stored-rebinned).sum()))

records={}
for coefficient,root in ROOTS.items():
    config=read(root/'config.json');result=read(root/'result.json');proof=read(root/'archive-verification.json')
    assert proof['passed'] and proof['steps']==result['steps']==1_000_000
    assert proof['training_source_commit']==config['source_commit']==result['source_commit']
    assert config['intrinsic']['coefficient']==coefficient and config['seed']==0
    with np.load(root/'resume/step_0001000000/replay.npz') as z:
        xy=z['next_observations'][:,:2].copy();oldxy=z['observations'][:,:2].copy()
        rewards=z['rewards'].copy()
    assert len(xy)==1_000_000 and np.isfinite(xy).all()
    distances=np.linalg.norm(xy[:,None,:]-np.asarray(config['environment']['goals'])[None,:,:],axis=2)
    np.testing.assert_allclose(rewards,-distances.min(1),rtol=2e-6,atol=1e-5)
    counts,ids=occupancy(xy)
    with np.load(root/'training_coverage.npz') as z:stored_counts=z['counts'].copy()
    coverage_audit=verify_coverage_rounding(xy,stored_counts,counts)
    assert np.array_equal(counts>0,stored_counts>0)
    counts=stored_counts
    episodes=read(root/'training_episodes.json')['episodes'];cursor=0;route_counts=Counter()
    goal_counts=Counter()
    for e in episodes:
        end=e['end_step'];assert end-cursor==e['length']
        if e['success']:
            path=np.concatenate([oldxy[cursor:cursor+1],xy[cursor:end]])
            route_counts[route_label('v3',path,e['goal_id'])]+=1
            goal_counts[str(e['goal_id'])]+=1
        cursor=end
    assert sum(goal_counts.values())==result['training']['training_successes']
    windows=[];cumulative=[];visited=set()
    for begin in range(0,1_000_000,100_000):
        end=begin+100_000;q=xy[begin:end];co,_=occupancy(q);visited.update(ids[begin:end].tolist())
        good=[e for e in episodes if begin<e['end_step']<=end]
        mass=co[co>0]/len(q)
        windows.append(dict(start=begin,end=end,visited_bins=int(np.count_nonzero(co)),
            g1_region_fraction=float(np.mean((q[:,0]<-2)&(q[:,1]>2))),
            g2_region_fraction=float(np.mean((q[:,0]>2)&(q[:,1]<-2))),
            min_goal_distances=distances[begin:end].min(0).tolist(),
            top10_fraction=float(np.sort(co.ravel())[-10:].sum()/len(q)),
            episodes=len(good),successes=sum(e['success'] for e in good)))
        cumulative.append({'step':end,'visited_bins':len(visited)})
    paths={};rollout={};input_hashes={}
    for mode in ['policy','native','zero_z']:
        file=root/'rollouts'/f'1000000-{mode}-fixed.npz'
        with np.load(file) as z:paths[mode]={k:z[k] for k in z.files}
        z=paths[mode];assert len(z['xy'])==100
        np.testing.assert_array_equal(z['initial_simulator_state'],np.broadcast_to(z['initial_simulator_state'][0],z['initial_simulator_state'].shape))
        s=summarize('v3',z['xy'],z['lengths'],z['goal_ids'],z['returns'])
        stored=result['summaries'][mode+'-fixed']
        assert s['routes']==stored['routes'] and s['success_rate']==stored['success_rate']
        for path,n,ret in zip(z['xy'],z['lengths'],z['returns']):
            d=np.linalg.norm(path[1:n+1,None,:]-np.asarray(config['environment']['goals'])[None,:,:],axis=2)
            np.testing.assert_allclose(ret,-d.min(1).sum(),rtol=2e-6,atol=.02)
        with np.load(root/'rollouts'/f'1000000-{mode}-natural.npz') as other:
            assert np.array_equal(z['xy'],other['xy'],equal_nan=True)
        rollout[mode]={k:v for k,v in stored.items() if k!='routes'}
        rollout[mode]['mean_episode_length']=float(z['lengths'].mean())
        rollout[mode]['failed_episode_indices']=np.flatnonzero(z['goal_ids']==0).tolist()
        input_hashes[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    prob=counts[counts>0]/counts.sum()
    summary=dict(coefficient=coefficient,training_source=config['source_commit'],seed=0,steps=1_000_000,
        visited_bins=int(np.count_nonzero(counts)),visited_area_m2=float(np.count_nonzero(counts)*.25),
        effective_bins=float(np.exp(-(prob*np.log(prob)).sum())),
        top10_fraction=float(np.sort(counts.ravel())[-10:].sum()/counts.sum()),
        min_goal_distances=distances.min(0).tolist(),
        g1_region_fraction=float(np.mean((xy[:,0]<-2)&(xy[:,1]>2))),
        g2_region_fraction=float(np.mean((xy[:,0]>2)&(xy[:,1]<-2))),
        first_success_step=result['training']['first_success_step'],training=result['training'],
        training_goal_counts=dict(goal_counts),training_route_counts=dict(route_counts),
        windows=windows,cumulative=cumulative,rollout=rollout,raw_rollout_sha256=input_hashes,
        natural_and_fixed_rollouts_identical=True,coverage_audit=coverage_audit)
    history={mode:read(root/f'history-{mode}-natural.json') for mode in ['policy','native']}
    records[coefficient]=dict(root=root,config=config,summary=summary,counts=counts,paths=paths,history=history)

initial=[read(r['root']/'parameter-audit.json')['initial'] for r in records.values()]
assert initial[0]==initial[1]
for mode in ['policy','native','zero_z']:
    for key in ['initial_simulator_state','env_seeds','policy_batch_seeds']:
        np.testing.assert_array_equal(records[.01]['paths'][mode][key],records[.1]['paths'][mode][key])
ca,cb=[json.loads(json.dumps(r['config'])) for r in records.values()]
for c in [ca,cb]:
    for k in ['profile','source_commit','source_root','eval_interval','checkpoint_interval']:c.pop(k)
    c['native'].pop('output_root');c['intrinsic'].pop('coefficient')
assert ca==cb

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def trajectories(selection,name,title):
    fig,axes=plt.subplots(1,len(selection),figsize=(6*len(selection),6),squeeze=False)
    for ax,(coef,mode) in zip(axes[0],selection):
        r=records[coef];maze(ax,r['config']);z=r['paths'][mode];s=r['summary']['rollout'][mode]
        order=np.argsort(z['goal_ids']==0)  # Draw every failure last and visibly.
        for i in order:
            path=z['xy'][i,:int(z['lengths'][i])+1];failed=z['goal_ids'][i]==0
            color='#ba5360' if failed else '#07877c'
            ax.plot(path[:,0],path[:,1],color=color,alpha=.9 if failed else .17,
                    lw=1.6 if failed else .8,zorder=5 if failed else 3)
            if failed:ax.scatter(*path[-1],color=color,marker='x',s=50,zorder=6)
        label='Direct policy: random z + sigma' if mode=='policy' else 'Mu-only: random z, no sigma'
        ax.set_title(f'NovelD {coef:g} | {label}\nSuccess {s["success_rate"]:.0%} | successful route categories: 1',fontsize=11)
    fig.suptitle(title+'\nActual saved trajectories · 1M interactions · seed 0 · 100 rollouts per panel',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color='#07877c',label='Reached G1'),Line2D([0],[0],color='#ba5360',label='Failed (endpoint ×)')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.007))
    fig.tight_layout(rect=(0,.045,1,.9));save(fig,name)

trajectories([(.1,'policy'),(.1,'native')],'final-trajectories','OptiQ v3: reaches G1 through one passage')
trajectories([(.01,'policy'),(.1,'policy')],'coefficient-trajectories','NovelD coefficient comparison: final direct policy')

fig,axes=plt.subplots(1,2,figsize=(12.6,6))
vmax=max(r['counts'].max() for r in records.values())
for ax,(coef,r) in zip(axes,records.items()):
    maze(ax,r['config']);co=r['counts'].astype(float);co[co==0]=np.nan
    im=ax.imshow(co.T,origin='lower',extent=(-18,18,-18,18),cmap='YlOrRd',norm=LogNorm(1,vmax),zorder=1)
    s=r['summary'];ax.set_title(f'NovelD {coef:g} | {s["visited_bins"]} visited bins\nClosest approach to G2: {s["min_goal_distances"][1]:.2f} m',fontsize=12)
fig.colorbar(im,ax=axes,fraction=.025,pad=.025,label='Training visits per 0.5 m × 0.5 m bin (log scale)')
fig.suptitle('Training exploration over 1M interactions\nSame grid and color scale · evaluation visits excluded · single seed 0',fontsize=14)
fig.subplots_adjust(top=.82,bottom=.1,wspace=.20,right=.86)
save(fig,'training-coverage')

fig,axes=plt.subplots(2,2,figsize=(12,8.5))
for coef,r in records.items():
    s=r['summary'];color=COLORS[coef];label=f'NovelD {coef:g}'
    axes[0,0].plot([0]+[x['step']/1e6 for x in s['cumulative']],[0]+[x['visited_bins'] for x in s['cumulative']],'-o',color=color,label=label,ms=4)
    w=s['windows'];axes[0,1].plot([x['end']/1e6 for x in w],[100*x['g2_region_fraction'] for x in w],'-o',color=color,label=label,ms=4)
    h=r['history']['policy'];axes[1,0].plot([x['step']/1e6 for x in h],[100*x['success_rate'] for x in h],'-o',color=color,label=label,ms=3)
    axes[1,1].plot([x['step']/1e6 for x in h],[x['mean_return'] for x in h],'-o',color=color,label=label,ms=3)
titles=['Cumulative training coverage','G2-side training visits per 100k window','Periodic direct-policy success (10 episodes)','Periodic direct-policy return (10 episodes)']
ylabels=['Visited 0.5 m bins','% training steps with x > 2, y < -2','Success (%)','Dense environment return']
for ax,title,ylabel in zip(axes.flat,titles,ylabels):
    ax.set(title=title,xlabel='Total environment interactions (M)',ylabel=ylabel);ax.grid(alpha=.18);ax.legend(frameon=False)
axes[1,0].set_ylim(-3,103)
fig.suptitle('v3 OptiQ: learning and exploration histories\n1 training seed; evaluation interval differs (25k vs 250k); no confidence bands',fontsize=14)
fig.tight_layout(rect=(0,0,1,.93));save(fig,'learning-and-exploration')

payload=dict(current=records[.1]['summary'],reference=records[.01]['summary'],
    common_model_initialization=True,common_environment_and_training_hyperparameters_except_coefficient=True,
    differences=['NovelD coefficient','evaluation interval 25k -> 250k','full-checkpoint interval 100k -> final only','source commits and output metadata'],
    limitation='One training seed. A geometric route category does not identify every possible within-corridor trajectory. Natural and fixed data are identical, not 200 independent trials.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
s=records[.1]['summary'];old=records[.01]['summary']
report=f'''# v3 OptiQ 1M 결과 분석

NovelD 0.1, seed0, 1M environment interactions / 995k learner updates. 학습 source `{s['training_source']}`.

## 최종 정책

- 직접 샘플링(random z + conditional sigma): 99/100 성공, 모두 G1/passage-y+4. 실패 1회도 그림에 포함.
- random z mu-only: 100/100 성공, 모두 같은 통로와 G1.
- z=0 mu-only: 100/100 성공, 동일 초기 상태에서 동일한 경로 반복.
- 목표 도달 성능은 높지만 여러 목표·통로를 사용하는 정책은 관찰되지 않았다. action distribution 자체가 단봉이라는 결론은 낼 수 없다.
- 실제 원시 좌표를 그렸다. 750k 도식과 달리 1M raw rollout이 보존되어 있다.

## 학습 중 탐색

- 최초 성공: {s['first_success_step']:,} steps. 전체 학습 성공: {s['training']['training_successes']:,}회, 목표별 {s['training_goal_counts']}.
- 학습 성공 경로 분류: {s['training_route_counts']}.
- 누적 방문 0.5m 격자: {s['visited_bins']}개. 이전 0.01은 {old['visited_bins']}개이며 현재가 {(1-s['visited_bins']/old['visited_bins'])*100:.1f}% 적다.
- G2 최단 접근 거리: 현재 {s['min_goal_distances'][1]:.2f}m / 이전 {old['min_goal_distances'][1]:.2f}m. 성공 기준은 0.5m.
- G2쪽 영역(x>2, y<-2) 방문 비율: 현재 {s['g2_region_fraction']:.3%}, 이전 {old['g2_region_fraction']:.3%}. 이는 이 영역의 시간 비율이며 전체 우측 공간 탐색이나 성공 경로 수와 동일하지 않다.
- 마지막 100k 방문 격자: 현재 {s['windows'][-1]['visited_bins']}, 이전 {old['windows'][-1]['visited_bins']}.

## 해석

이 한 seed에서는 계수 0.01에서 0.1로 높여도 최종 목표·경로 다양성이 늘지 않았고, 전체 공간 탐색 범위는 줄었다. 직접 샘플링 최종 성공률은 두 조건 모두 99%였다.
G1 또는 G2 중 가까운 목표에 도달하면 되는 보상이며, NovelD는 목표별 방문 균형을 직접 최적화하지 않는다. 관찰 결과는 G1 통로에 집중된 학습과 일치하지만, 원인을 계수 하나로 확정하려면 추가 seed/통제 실험이 필요하다. CPU/GPU 병렬화는 속도 개선이며 다중 경로를 자동 보장하지 않는다.

## 비교·검증 한계

이전 학습 source `{old['training_source']}`. 모델 초기 actor/critic hash, 환경, 나머지 학습 config는 일치한다. 계수 외에도 평가 주기(25k 대 250k), full checkpoint 주기(100k 대 최종만), source commit이 다르다. 이번 최종100k에는 정기 평가가 1M 한 점뿐이므로 촘촘한 마지막100k 평가 평균을 주장하지 않는다.
두 reset 모드의 초기 상태 및 rollout 배열이 동일하여 natural/fixed를 합쳐 200회로 세지 않는다. 결과는 각각 단일 training seed0 정책의 100회 평가다.
기존 archive SHA256/full-state 검증을 확인하고, 1M replay 보상을 목표 거리와 재대조했다. Coverage는 float64 시뮬레이터 좌표, replay는 float32 관측이므로 각 조건에서 경계상의 1개 표본이 인접 격자로 반올림된다. 모든 차이가 float32 반올림 구간으로 설명됨을 검증했고 방문 격자 집합은 완전히 일치했다. 방문 지도는 원래 저장 coverage를 사용한다. Raw rollout에서 성공률·경로·보상을 재계산하여 저장 summary와 대조했다. 입력 rollout hash는 results.json에 기록했다.

그림: final-trajectories.png, coefficient-trajectories.png, training-coverage.png, learning-and-exploration.png. PDF도 보관.
'''
(OUT/'REPORT_KO.md').write_text(report)
print(json.dumps({coef:{k:r['summary'][k] for k in ['visited_bins','training_goal_counts','training_route_counts','min_goal_distances','g2_region_fraction']} for coef,r in records.items()},indent=2))
print(OUT)
