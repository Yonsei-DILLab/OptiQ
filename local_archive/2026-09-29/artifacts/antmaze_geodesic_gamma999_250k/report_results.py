"""Post-hoc reward-validated comparisons; never change a policy or raw file."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent
WORKTREE = ARTIFACTS.parent/'tmp/reward-progress-worktree'
SOURCE = '438907f3a5ef681d6cde9012a66c7336fa642546'
CONTROL_SOURCE = 'eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45'
OUT = ROOT/'report'
TASKS = ('v1', 'v3', 'v4')
MODES = ('policy', 'native')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper_path = ARTIFACTS/'antmaze_horizon_temperature_250k/report_results.py'
helper = module('route_helpers', helper_path)
reward_path = WORKTREE/'antmaze_experiments/progress_reward.py'
reward_bytes = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
                                       SOURCE+':antmaze_experiments/progress_reward.py'])
# Execute the exact historical implementation even if a later experiment adds
# a new reward profile to the active worktree. Geometry must still match it.
reward = ModuleType('verified_progress_reward')
reward.__file__ = str(reward_path)
exec(compile(reward_bytes, str(reward_path), 'exec'), reward.__dict__)
upstream_bytes = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
    SOURCE+':antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py'])
assert reward.UPSTREAM.read_bytes() == upstream_bytes


def read(path):
    return json.loads(path.read_text())


def evaluate(path, cfg, condition, mode):
    s = read(path)
    raw = path.with_name('rollouts.npz')
    with np.load(raw, allow_pickle=False) as z:
        assert str(z['mode']) == mode and int(z['env_steps']) == s['step']
        points = [p[:int(k)+1].copy() for p, k in zip(z['xy'], z['lengths'])]
        goals, returns, starts = z['goals'].copy(), z['returns'].copy(), z['initial_full_state'].copy()
    task, n = cfg['task'], len(points)
    assert n == s['episodes'] == len(goals) == len(returns)
    assert np.isclose((goals > 0).mean(), s['success_rate'])
    assert np.isclose(returns.mean(), s['mean_return'])
    assert all(np.isfinite(p).all() for p in points)
    if task == 'v1':
        assert not s['fixed'] and not s['identical_initial_full_state']
    else:
        assert s['original_origin_fixed'] and s['identical_initial_full_state']
        np.testing.assert_array_equal(starts, np.repeat(starts[:1], n, axis=0))
        assert np.all(starts[:, :2] == 0)
    profile = cfg['reward_profile']
    assert cfg['reward_specification'] == reward.specification(task, profile)
    endpoints = np.asarray([[p[0], p[-1]] for p in points])
    d = reward.distance(endpoints, task, profile)
    np.testing.assert_allclose(returns, 100*(d[:, 0]-d[:, 1]), atol=.002, rtol=2e-5)
    _, goal_xy, _ = reward.maze_geometry(task)
    euclidean = [np.linalg.norm(p[:, None]-goal_xy, axis=-1).min(-1) for p in points]
    for p, goal in zip(points, goals):
        if goal > 0:
            assert np.linalg.norm(p[-1]-goal_xy[int(goal)-1]) < .501
    labels = [helper.first_gate(task, p) for p in points]
    entries = dict(Counter(labels))
    successes = dict(Counter(label for label, goal in zip(labels, goals) if goal > 0))
    sides = ('left', 'right') if task == 'v3' else ('upper', 'lower')
    minority = min(successes.get(side, 0) for side in sides)
    row = dict(task=task, condition=condition, mode=mode, id=path.parents[3].name,
        step=s['step'], episodes=n, source_commit=cfg['source_commit'],
        route_counts=entries, successful_route_counts=successes,
        successful_goal_routes=dict(Counter(f'{label}/G{goal}' for label, goal in zip(labels, goals) if goal > 0)),
        success_rate=float((goals>0).mean()), mean_return=float(returns.mean()),
        minority_entry_rate=min(entries.get(side, 0) for side in sides)/n,
        minority_success_rate=minority/n,
        screen_both_successful_routes_ge10pct=minority>=max(4,.1*n),
        closest_euclidean_goal_distance_mean=float(np.mean([x.min() for x in euclidean])),
        final_reward_distance_mean=float(d[:,1].mean()),
        fixed_original_start=task!='v1', reward_telescope_verified=True,
        raw_path=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    return dict(row=row,points=points,goals=goals,labels=labels)


def collect():
    runs = {}
    for condition, campaign, source in (
        ('euclidean', ARTIFACTS/'antmaze_horizon_temperature_250k', CONTROL_SOURCE),
        ('geodesic', ROOT, SOURCE)):
        for host in (campaign/'results').glob('vast-heechan-*'):
            manifest = read(host/'manifest.json')
            assert manifest['source_commit'] == source
            for job in manifest['jobs']:
                if job['task'] not in TASKS or job['hypothesis'] not in ('gamma999','geodesic_gamma999'):
                    continue
                folder=host/'runs'/job['id']
                key=job['task'],condition
                assert key not in runs
                run=dict(job=job,host=host.name,evaluations={mode:[] for mode in MODES},verified_final=False)
                runs[key]=run
                if not (folder/'config.json').exists():continue
                cfg=read(folder/'config.json')
                assert cfg['source_commit']==source and cfg['seed']==0
                assert cfg['native']['alg']['gamma']==.999 and cfg['temperature']==1
                assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
                assert cfg['dacer_enabled'] and not cfg['noveld_enabled'] and cfg['eval_starts']=='upstream'
                expected=reward.NO_COST_PROFILE if condition=='geodesic' else reward.EUCLIDEAN_NO_COST_PROFILE
                assert cfg['reward_profile']==expected
                reset='natural' if job['task']=='v1' else 'fixed'
                for mode in MODES:
                    for path in sorted((folder/'evaluations').glob(f'*/{mode}-{reset}/summary.json')):
                        if path.with_name('rollouts.npz').exists():
                            run['evaluations'][mode].append(evaluate(path,cfg,condition,mode))
                if (folder/'result.json').exists():
                    result=read(folder/'result.json')
                    assert result['completed'] and result['source_commit']==source
                    assert result['steps']==258304 and result['updates']==7816
                    assert result['checkpoint']['environment_reward_verified']
                    run['verified_final']=True
    return runs


def draw(ax, task, evaluation, title):
    walls,goals,bounds=reward.maze_geometry(task)
    for x0,y0,x1,y1 in walls:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e2e6e9',edgecolor='#bdc4cb',lw=.4))
    if evaluation:
        r=evaluation['row']
        for i in np.argsort(evaluation['goals']>0):
            p=evaluation['points'][i];success=evaluation['goals'][i]>0
            ax.plot(p[:,0],p[:,1],color=helper.ROUTE_COLORS[evaluation['labels'][i]],alpha=.5 if success else .2,lw=1 if success else .7)
        entries=' '.join(f'{helper.SHORT[k]}:{v}' for k,v in r['route_counts'].items())
        successes=' '.join(f'{helper.SHORT[k]}:{v}' for k,v in r['successful_route_counts'].items()) or 'none'
        title+=f"\n{r['step']/1000:.1f}k | n={r['episodes']} | entry {entries}\nsuccessful routes {successes}"
    else:title+='\nNo matched raw evaluation available'
    ax.set_title(title,fontsize=9)
    ax.scatter(*goals.T,marker='*',s=85,c='#38a35f',edgecolor='white',lw=.5)
    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')


def main():
    OUT.mkdir(exist_ok=True)
    runs=collect()
    history=[e['row'] for run in runs.values() for es in run['evaluations'].values() for e in es]
    latest=[es[-1]['row'] for run in runs.values() for es in run['evaluations'].values() if es]
    matched=[]
    for mode in MODES:
        fig,axes=plt.subplots(3,2,figsize=(10,13),layout='constrained')
        for row,task in enumerate(TASKS):
            pairs={c:{e['row']['step']:e for e in runs.get((task,c),{}).get('evaluations',{}).get(mode,[])} for c in ('euclidean','geodesic')}
            common=sorted(pairs['euclidean'].keys()&pairs['geodesic'].keys())
            step=common[-1] if common else None
            for col,c in enumerate(pairs):
                e=pairs[c].get(step)
                draw(axes[row,col],task,e,f'{task} | {c}')
                if e:matched.append(e['row'])
        fig.suptitle(f'Reward-only comparison | gamma .999, T1, DACER H/d+.7 | {mode}\nSame budget within each row; seed0. No external DACER noise.\nv1 native random reset; v3/v4 original fixed full state.',fontsize=11)
        fig.savefig(OUT/f'matched_trajectories_{mode}.png',dpi=155);plt.close(fig)
    payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),goal_achieved=False,
        source_commit=SOURCE,control_source=CONTROL_SOURCE,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        reward_implementation_sha256=hashlib.sha256(reward_bytes).hexdigest(),
        final_verified_jobs=[run['job']['id'] for key,run in runs.items() if key[1]=='geodesic' and run['verified_final']],
        latest=latest,history=history,exact_step_matched=matched,
        limitations=['All training seed0. Successful entries are not sustained goal-reaching.',
            'Geodesic reward is XY point-agent distance, not full-body configuration-space distance.',
            'v3 is geometrically asymmetric. v1 random-start counts are not same-state multimodality.',
            'Missing199 control raw files are unavailable, not assumed failed or zero success.'])
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Geodesic 거리만 바꾼 gamma .999 비교','',
        '동일 seed0, gamma .999, T1, DACER 목표 +.7/차원·500update 간격입니다. 알고리즘 변경 없이 기존 거리 보상만 비교합니다.','',
        '| 환경 | 거리 | 평가 | step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |',
        '|---|---|---|---:|---:|---|---|---:|']
    for r in sorted(latest,key=lambda x:(x['task'],x['condition'],x['mode'])):
        lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|{r['success_rate']:.1%}|")
    lines+=['','![동일 step 직접 정책](matched_trajectories_policy.png)','![mu-only 보조](matched_trajectories_native.png)','',
        '각 그림 행은 정확히 동일한 step만 비교합니다. 위 표는 가장 최근 저장 평가이므로 step이 다르면 직접 성능 비교를 하지 않습니다. '
        '반대 통로 방문과 그 경로로 실제 성공하는 것을 구분합니다. v1은 학습과 같은 랜덤 시작이며, v3/v4는 원래 고정 full state입니다. '
        '학습 source·원시 궤적 SHA256·실제 거리 보상 telescope를 검증하고 학습 자료는 변경하지 않았습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT),latest=latest,verified_final=payload['final_verified_jobs']),indent=2))


if __name__=='__main__':main()
