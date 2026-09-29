"""Verify same-state supplementary rollouts without replacing v1 primary data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = '8e9d7d3c2c79f797654ccfb21913d2c000b89f71'
EVAL_SOURCE = '2617549326b25f3dd00987e06d75de6bdb5e685b'
HELPER = ROOT.parent/'antmaze_geodesic_gamma999_250k/report_results.py'
spec = importlib.util.spec_from_file_location('geodesic_origin_helper', HELPER)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
OUT = ROOT/'report'


def read(path):
    return json.loads(path.read_text())


def collect():
    records = []
    for folder in sorted((ROOT/'results/vast-heechan-180').glob('step*')):
        if not (folder/'result.json').exists():
            continue
        result, proof, provenance = (read(folder/(name+'.json')) for name in ('result','verification','provenance'))
        assert result['completed'] and proof['passed']
        assert proof['training_checkpoint_unchanged'] and proof['model_optimizer_unchanged']
        assert proof['restored_policy_exact'] and proof['identical_initial_full_state']
        assert provenance['training_source'] == result['training_source'] == SOURCE
        assert provenance['evaluation_source'] == result['evaluation_source'] == EVAL_SOURCE
        assert provenance['supplementary_origin_probe'] and not provenance['replaces_primary_evaluation']
        assert not provenance['evaluation_matches_training_resets']
        assert provenance['inference_only'] and not provenance['external_noise'] and not provenance['intrinsic_reward']
        cfg = provenance['training_config']; step = result['step']
        assert cfg['task'] == 'v1' and cfg['seed'] == 0
        assert cfg['reward_profile'] == base.reward.NO_COST_PROFILE
        assert cfg['reward_specification'] == base.reward.specification('v1', cfg['reward_profile'])
        starts_by_mode = []
        for mode in ('policy', 'native'):
            path = folder/'evaluations'/f'{step:010d}'/(mode+'-fixed')/'rollouts.npz'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == proof['raw_sha256'][str(path.relative_to(folder))]
            s = read(path.with_name('summary.json'))
            with np.load(path, allow_pickle=False) as data:
                assert str(data['mode']) == mode and int(data['env_steps']) == step
                points = [p[:int(n)+1].copy() for p, n in zip(data['xy'], data['lengths'])]
                goals, returns, starts = data['goals'].copy(), data['returns'].copy(), data['initial_full_state'].copy()
            assert len(points) == s['episodes'] == provenance['episodes_per_mode'] == 100
            assert s['original_origin_fixed'] and s['identical_initial_full_state']
            np.testing.assert_array_equal(starts, np.repeat(starts[:1],100,axis=0))
            np.testing.assert_array_equal(starts[:,:2], np.zeros((100,2)))
            starts_by_mode.append(starts)
            assert all(np.isfinite(p).all() for p in points)
            endpoints = np.asarray([[p[0],p[-1]] for p in points])
            d = base.reward.distance(endpoints,'v1',cfg['reward_profile'])
            np.testing.assert_allclose(returns,100*(d[:,0]-d[:,1]),atol=.002,rtol=2e-5)
            assert np.isclose((goals>0).mean(),s['success_rate'])
            assert np.isclose(returns.mean(),s['mean_return'])
            _, target, _ = base.reward.maze_geometry('v1')
            for p, goal in zip(points, goals):
                if goal>0:
                    assert np.linalg.norm(p[-1]-target[int(goal)-1])<.501
            labels = [base.helper.first_gate('v1',p) for p in points]
            row = dict(task='v1',step=step,mode=mode,episodes=100,
                route_counts=dict(Counter(labels)),
                successful_route_counts=dict(Counter(l for l,g in zip(labels,goals) if g>0)),
                failures=int((goals==0).sum()),success_rate=float((goals>0).mean()),
                raw_sha256=digest,checkpoint_sha256=result['checkpoint_sha256'],
                training_source=SOURCE,evaluation_source=EVAL_SOURCE,
                identical_origin_verified=True,reward_telescope_verified=True,supplementary=True)
            records.append(dict(row=row,points=points,goals=goals,labels=labels))
        np.testing.assert_array_equal(*starts_by_mode)
    return records


def main():
    OUT.mkdir(exist_ok=True)
    records = collect()
    steps = sorted({r['row']['step'] for r in records})
    if steps:
        fig, axes = plt.subplots(len(steps),2,figsize=(10,4.8*len(steps)),squeeze=False,layout='constrained')
        for i, step in enumerate(steps):
            for j, mode in enumerate(('policy','native')):
                entry = next(r for r in records if r['row']['step']==step and r['row']['mode']==mode)
                base.draw(axes[i,j],'v1',entry,f'v1 geodesic | {mode} | identical origin')
        fig.suptitle('Supplement only: same position, pose and velocity; one training seed\n'
                     '100 episodes per mode/checkpoint. Primary random-start evaluation is preserved.',fontsize=11)
        fig.savefig(OUT/'same_state_routes.png',dpi=155);plt.close(fig)
    rows = [r['row'] for r in records]
    (OUT/'results.json').write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
        training_source=SOURCE,evaluation_source=EVAL_SOURCE,rows=rows,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        primary_random_evaluation_preserved=True,all_maze_goal_achieved=False),indent=2)+'\n')
    lines=['# v1 geodesic 정책의 같은 시작 상태 평가','',
        '원래 랜덤 시작 평가는 그대로 두고, 같은 중앙 위치·자세·속도에서 반복한 보조 검증입니다. '
        '조건부 sigma를 포함한 직접 정책과 mu-only를 분리하며, 외부 DACER 행동잡음은 넣지 않습니다.','',
        '| step | 평가 | 횟수 | 통로 진입 | 성공 통로 | 실패 | 성공률 |','|---:|---|---:|---|---|---:|---:|']
    for r in rows:
        lines.append(f"|{r['step']}|{r['mode']}|100|{r['route_counts']}|{r['successful_route_counts']}|{r['failures']}|{r['success_rate']:.1%}|")
    lines+=['','![같은 상태 궤적](same_state_routes.png)','',
        '학습 seed0 하나, 선택한 중앙 상태 하나입니다. 체크포인트 간 유지 여부는 후속 시점을 함께 확인해야 합니다. '
        '모든 원시 파일 SHA256·full state·보상 합·모델/체크포인트 불변을 검증했습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':rows,'report':str(OUT)}))


if __name__=='__main__':
    main()
