"""Independent raw-file verification of the paired inference-only horizon test."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = 'db4ca0a446f5fe8df1dda02e261fa9154c66d9c9'
EVAL_SOURCE = 'ed45db5c0d5d27c468387b8f36ce92acbf0324ac'
W = ROOT.parent.parent / 'tmp/reward-progress-worktree'
HELPER = ROOT.parent / 'antmaze_startnorm_geodesic_250k/report_results.py'
spec = importlib.util.spec_from_file_location('verified_horizon_report', HELPER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base, reward = m.base, m.reward
diag_bytes = subprocess.check_output(['git', '-C', str(W), 'show', SOURCE + ':antmaze_experiments/critic_diagnostics.py'])
diag = ModuleType('frozen_route_labels')
exec(compile(diag_bytes, 'frozen_critic_diagnostics.py', 'exec'), diag.__dict__)


def read(p):
    return json.loads(p.read_text())


def main():
    folder = ROOT / 'results/vast-heechan-180/evaluation'
    if not (folder / 'result.json').exists():
        print(json.dumps(dict(completed=False, reason='Paired inference is not complete')))
        return
    result, proof, provenance = [read(folder / (n + '.json')) for n in ('result', 'verification', 'provenance')]
    assert result['completed'] and result['inference_only'] and proof['passed']
    assert result['training_source'] == provenance['training_source'] == SOURCE
    assert result['evaluation_source'] == provenance['evaluation_source'] == EVAL_SOURCE
    for flag in ('identical_initial_full_state', 'paired_prefix_exact', 'checkpoint_unchanged',
                 'model_optimizer_unchanged', 'evaluation_rng_restored', 'primary_evaluation_unchanged'):
        assert proof[flag]
    cfg = provenance['training_config']
    assert cfg['task'] == 'v3' and cfg['temperature'] == 3. and cfg['discount'] == .999
    assert cfg['reward_profile'] == reward.START_NORMALIZED_PROFILE
    assert cfg['reward_specification'] == reward.specification('v3', cfg['reward_profile'])
    assert not provenance['external_dacer_noise'] and provenance['conditional_sigma'] and provenance['random_latent']
    assert provenance['native_limit'] == 700 and provenance['extended_limit'] == 1400
    _, targets, _ = reward.maze_geometry('v3')
    conditions = {}
    for limit in (700, 1400):
        raw = folder / f'rollouts-limit{limit}.npz'
        summary = result['conditions'][str(limit)]
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest == summary['raw_sha256']
        z = np.load(raw)
        assert str(z['mode']) == 'policy' and bool(z['fixed']) and int(z['episode_limit']) == limit
        assert int(z['env_steps']) == provenance['checkpoint_step'] == 258304
        points = [p[:int(n) + 1].copy() for p, n in zip(z['xy'], z['lengths'])]
        starts, goals, lengths, returns = [z[k].copy() for k in ('initial_full_state', 'goals', 'lengths', 'returns')]
        assert len(points) == summary['episodes'] == 100
        np.testing.assert_array_equal(starts, np.repeat(starts[:1], 100, axis=0))
        np.testing.assert_array_equal(starts[:, :2], np.zeros((100, 2)))
        assert all(np.isfinite(p).all() for p in points) and np.isfinite(returns).all()
        assert np.all((lengths > 0) & (lengths <= limit))
        endpoints = np.array([[p[0], p[-1]] for p in points])
        d = reward.distance(endpoints, 'v3', cfg['reward_profile'])
        np.testing.assert_allclose(returns, 100 * (d[:, 0] - d[:, 1]), atol=.002, rtol=2e-5)
        for p, g in zip(points, goals):
            distances = np.linalg.norm(p[:, None, :] - targets[None, :, :], axis=-1)
            if g:
                assert distances[-1, int(g) - 1] < .501
            else:
                assert distances.min() >= .499
        labels = [diag.route_label('v3', p)[0] for p in points]
        counts = dict(Counter(labels))
        successes = dict(Counter(label for label, g in zip(labels, goals) if g))
        assert counts == summary['routes'] and successes == summary['successful_routes']
        row = dict(step=258304, episodes=100, limit=limit, mode='policy',
            route_counts=counts, successful_route_counts=successes,
            successful_goal_routes=dict(Counter(f'{label}/G{g}' for label, g in zip(labels, goals) if g)),
            failures=int((goals == 0).sum()), raw_sha256=digest,
            supplementary=limit != 700, mean_length=float(lengths.mean()))
        conditions[limit] = dict(row=row, points=points, labels=labels, goals=goals,
                                 starts=starts, lengths=lengths)
    a, b = conditions[700], conditions[1400]
    np.testing.assert_array_equal(a['starts'], b['starts'])
    for i, (p, q) in enumerate(zip(a['points'], b['points'])):
        np.testing.assert_array_equal(p, q[:len(p)])
        if a['goals'][i]:
            assert b['goals'][i] == a['goals'][i] and b['lengths'][i] == a['lengths'][i]
        else:
            assert a['lengths'][i] == 700
    extra = int(((a['goals'] == 0) & (b['goals'] > 0)).sum())
    out = ROOT / 'report'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), layout='constrained')
    for ax, limit in zip(axes, (700, 1400)):
        title = 'Native 700-step reference' if limit == 700 else 'SUPPLEMENT: 1400-step limit'
        base.draw(ax, 'v3', conditions[limit], title)
    fig.suptitle('Same frozen v3 policy; 100 paired CPU direct-policy episodes; identical full start\n'
                 'Exact native-length trajectory prefixes; extra-time successes are NOT native-budget successes', fontsize=10)
    fig.savefig(out / 'paired_horizon_trajectories.png', dpi=160)
    plt.close(fig)
    rows = [conditions[n]['row'] for n in (700, 1400)]
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), training_source=SOURCE,
        evaluation_source=EVAL_SOURCE, raw_prefix_verification=True, extra_time_successes=extra,
        rows=rows, goal_achieved=False, primary_evaluation_preserved=True,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# 저장된 v3 정책의 시간 제한 보조 진단', '',
        '학습을 바꾸지 않고 같은 체크포인트·시작 상태·난수로 평가했습니다. '
        '1,400-step 결과는 보조 진단이며 원래 700-step 성공률에 합치지 않습니다.', '',
        '| 평가 제한 | 전체 | 진입 경로 | 성공 경로 | 실패 |', '|---:|---:|---|---|---:|']
    for r in rows:
        lines.append(f"|{r['limit']}|100|{r['route_counts']}|{r['successful_route_counts']}|{r['failures']}|")
    lines.extend(['', f'700-step에서는 실패했지만 추가 시간에 성공한 에피소드: {extra}/100.', '',
        '![같은 정책의 paired 궤적](paired_horizon_trajectories.png)', '',
        '원래 길이까지 모든 paired 궤적이 정확히 일치함을 원자료로 재검증했습니다. '
        '단일 학습 seed의 보조 분석이며, 학습 시간 제한을 바꾼 효과나 장기 경로 유지를 증명하지 않습니다.'])
    (out / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(rows=rows, extra_time_successes=extra, report=str(out))))


if __name__ == '__main__':
    main()
