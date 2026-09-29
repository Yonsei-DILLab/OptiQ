"""Verify immutable official v1 policies, same-state inference and retention."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parent
WORKTREE=ROOT.parent.parent/'tmp/reward-progress-worktree'
sys.path.insert(0,str(WORKTREE))
from antmaze_experiments.critic_diagnostics import route_label
from antmaze_experiments.progress_reward import maze_geometry
from antmaze_experiments.register_horizon_temperature import CORE_FILES
TRAINING='5baa5b3463416cdfdcad465e8f862f3729802568'
REPORTING='5c3c966f2bfdc83242b449534712c65fd4c7ac6e'
REVISED_REPORTING='ae556841753c5baa1d58a227e8cc90fbf4aa129b'
NAMES={'step2m':'2M checkpoint','step2m5':'2.5M checkpoint',
       'step2m75-r2':'2.75M checkpoint',
       'final-3m-seed20260925':'3.008M final','final-independent':'3.008M final, new eval RNG'}
COLORS={'upper':'#277db5','lower':'#e58a27','uncommitted':'#8d959e'}
OUT=ROOT/'report'


def read(p):return json.loads(p.read_text())


def wilson(k,n):
    p=k/n;z=1.96;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-radius),float(mid+radius)]


def collect():
    evaluations={};configs={}
    for name in NAMES:
        folder=ROOT/'results'/name
        if not (folder/'verification.json').exists() or not (folder/'result.json').exists():continue
        p=read(folder/'provenance.json');proof=read(folder/'verification.json')
        expected_reporting=REVISED_REPORTING if name=='step2m75-r2' else REPORTING
        assert p['training_source']==TRAINING and p['evaluation_source']==expected_reporting
        assert p['supplementary_origin_probe'] and not p['replaces_primary_evaluation']
        assert p['external_exploration_noise'] is False and p['intrinsic_reward'] is False
        assert proof['passed'] and proof['checkpoint_unchanged'] and proof['model_optimizer_unchanged']
        assert proof['paired_identical_origin_states'] and proof['goals_and_dense_returns_recomputed']
        cfg=p['training_config'];configs[name]=cfg
        assert cfg['method']=='optiq' and cfg['task']=='v1' and cfg['reward_profile']=='dense'
        assert cfg['temperature']==3 and cfg['dacer_enabled'] is False and cfg['noveld_enabled'] is False
        states={}
        for mode in ('policy','native'):
            raw=folder/(mode+'.npz')
            assert hashlib.sha256(raw.read_bytes()).hexdigest()==proof['npz_sha256'][raw.name]
            with np.load(raw,allow_pickle=False) as z:
                goals=z['goal_ids'].copy();returns=z['returns'].copy();lengths=z['lengths'].copy()
                states[mode]=z['initial_full_state'].copy()
                points=[x[:int(n)+1].copy() for x,n in zip(z['xy'],lengths)]
                if 'closest_physics_distance' in z:
                    np.testing.assert_array_equal(z['closest_physics_distance']<=.5,goals>0)
            n=len(points);assert n==100
            np.testing.assert_array_equal(states[mode],np.repeat(states[mode][:1],n,axis=0))
            assert np.all(states[mode][:,:2]==0)
            _,goal_xy,_=maze_geometry('v1')
            distances=[np.linalg.norm(x[:,None]-goal_xy,axis=-1).min(-1) for x in points]
            np.testing.assert_allclose(returns,[-d[1:].sum() for d in distances],atol=.004,rtol=2e-6)
            for x,g in zip(points,goals):
                assert np.isfinite(x).all()
                if g>0:assert np.linalg.norm(x[-1]-goal_xy[int(g)-1])<.501
            labels=[route_label('v1',x)[0] for x in points]
            entries=dict(Counter(labels))
            success=dict(Counter(l for l,g in zip(labels,goals) if g>0))
            row=dict(name=name,label=NAMES[name],mode=mode,step=p['checkpoint_steps'],
                evaluation_source=p['evaluation_source'],
                episodes=n,training_seed=p['training_seed'],evaluation_seed=p['seed_sequence'],
                success_rate=float((goals>0).mean()),failures=int((goals==0).sum()),
                entries=entries,successful_routes=success,
                successful_route_wilson95={s:wilson(success.get(s,0),n) for s in ('upper','lower')},
                full_initial_state=states[mode][0].tolist(),
                checkpoint_sha256=p['checkpoint_sha256'],raw_path=str(raw),raw_sha256=proof['npz_sha256'][raw.name],
                reward_goal_and_full_state_verified=True,checkpoint_and_parameters_unchanged=True)
            evaluations[name,mode]=dict(row=row,points=points,goals=goals,labels=labels)
        np.testing.assert_array_equal(states['policy'],states['native'])
    return evaluations,configs


def main():
    OUT.mkdir(exist_ok=True)
    evaluations,configs=collect()
    rows=[e['row'] for e in evaluations.values()]
    core=[]
    for path in CORE_FILES:
        old=subprocess.check_output(['git','-C',str(WORKTREE),'show',TRAINING+':'+path])
        new=subprocess.check_output(['git','-C',str(WORKTREE),'show',REPORTING+':'+path])
        core.append(dict(path=path,training_sha256=hashlib.sha256(old).hexdigest(),
                         reporting_sha256=hashlib.sha256(new).hexdigest(),identical=old==new))
    # Only regulator instrumentation differs; this historical policy disables DACER.
    assert all(r['identical'] for r in core if not r['path'].endswith('/regulator.py'))
    complete=len(evaluations)==2*len(NAMES)
    retained=complete and all(min(e['row']['successful_routes'].get(s,0) for s in ('upper','lower'))>=10
                             for e in evaluations.values())
    late_names=('step2m75-r2','final-3m-seed20260925','final-independent')
    late_retained=all((name,mode) in evaluations and
        min(evaluations[name,mode]['row']['successful_routes'].get(s,0) for s in ('upper','lower'))>=10
        for name in late_names for mode in ('policy','native'))
    payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),training_source=TRAINING,
        evaluation_sources=[REPORTING,REVISED_REPORTING],report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,all_probes_verified=complete,
        both_successful_routes_ge10pct_all_probes=retained,algorithm_file_comparison=core,
        late_2m75_to_final_window_ge10pct=late_retained,
        late_window_scope='Post-hoc exploratory late window; earlier2M/2.5M failures remain in the full table.',
        claim_scope='Official v1, one trained seed, identical central start; trajectory multimodality, not action-density proof or all-maze guarantee.',
        limitations=['Supplementary central-start evaluation; primary v1 random-start data are preserved.',
            'All checkpoints belong to one training seed; independent evaluation RNG is not an independent training seed.',
            'Different starting positions may prefer one route. No uniform multimodality claim across the state space.',
            'This is historical dense=-distance/T3/DACER-off, not the new entropy/progress-reward sweep.'])
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    walls,goals,bounds=maze_geometry('v1')
    fig,axes=plt.subplots(2,len(NAMES),figsize=(4*len(NAMES),9),layout='constrained')
    for col,name in enumerate(NAMES):
        for row,mode in enumerate(('policy','native')):
            ax=axes[row,col]
            for x0,y0,x1,y1 in walls:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e2e6e9',edgecolor='#bdc4cb',lw=.4))
            e=evaluations.get((name,mode))
            title=NAMES[name]+'\n'+('random z + sigma' if mode=='policy' else 'random-z mu-only')
            if e:
                r=e['row']
                for i in np.argsort(e['goals']>0):
                    x=e['points'][i];ok=e['goals'][i]>0
                    ax.plot(x[:,0],x[:,1],color=COLORS[e['labels'][i]],alpha=.35 if ok else .12,lw=.9 if ok else .6)
                title+=f"\nsuccess U:{r['successful_routes'].get('upper',0)} D:{r['successful_routes'].get('lower',0)}; failed:{r['failures']}"
            else:title+='\nPending verified raw data'
            ax.set_title(title,fontsize=9)
            ax.scatter(*goals.T,marker='*',s=90,c='#38a35f',edgecolor='white',lw=.5)
            ax.scatter(0,0,c='black',s=18,zorder=6)
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
            ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Official v1: successful routes from the same complete initial state\nHistorical T3, dense=-distance, DACER OFF. n=100 per panel; all training seed0.\nSupplementary central-start probe; original random-start evaluation is preserved.',fontsize=12)
    fig.savefig(OUT/'same_state_retention.png',dpi=165);plt.close(fig)
    lines=['# 공식 v1 동일 시작 상태에서의 성공 경로 유지','',
        '과거 T3, dense=-거리, DACER OFF 정책을 수정하지 않고 재평가했습니다. 원점에서 위치·자세·속도까지 모두 같은 상태로 시작하며, 최종 정책은 새로운 평가 난수로도 다시 확인합니다. v1의 원래 랜덤 시작 평가는 보존된 별도 주 평가입니다.','',
        '| 체크포인트 | 평가 | 위쪽 성공 | 아래쪽 성공 | 실패 |',
        '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['label']}|{r['mode']}|{r['successful_routes'].get('upper',0)}|{r['successful_routes'].get('lower',0)}|{r['failures']}|")
    lines+=['','![동일 상태 경로 유지](same_state_retention.png)','',
        '각 행은100회입니다. policy는 random z+conditional sigma이고 native는 random-z mu-only입니다. 외부 DACER 행동잡음, 학습 업데이트, best-of 선택은 없습니다. '
        '전체 초기 상태 동일성, 보상 거리합, 실제 goal 반경, 원시 데이터 SHA256, 파라미터와 원본 체크포인트 보존을 검증했습니다.','',
        '이 결과의 범위는 공식 v1에서 하나의 학습 seed 정책이 같은 시작 상태에서도 두 성공 경로를 생성하는지입니다. '
        '행동 밀도 자체의 다봉성, 모든 시작 상태, v3/v4, 여러 학습 seed의 재현성을 자동으로 입증하지 않습니다. '
        '서로 다른 checkpoint나 평가 모드를 섞어 하나의 정책처럼 집계하지 않았습니다.','',
        f'학습 source `{TRAINING}`, 별도 평가 source `{REPORTING}` 및2.75M 반경 검증 보정 `{REVISED_REPORTING}`. actor/critic/TD/NLL/sampler 핵심6개 파일은 현재 소스와 바이트 단위로 같습니다. '
        'regulator 차이는 진단 기록 추가이며 해당 과거 정책에서는 DACER가 꺼져 있습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=rows,all_probes=complete,retained=retained),indent=2))


if __name__=='__main__':main()
