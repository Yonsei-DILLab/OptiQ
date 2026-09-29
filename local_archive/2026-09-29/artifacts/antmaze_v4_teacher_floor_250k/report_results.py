"""Post-hoc v4 teacher-floor comparison; preserves every training/raw artifact."""
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

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent
WORKTREE = ARTIFACTS.parent / 'tmp/reward-progress-worktree'
SOURCE = '555bb7e3c0101ee939545cc35d4d0729291781ec'
CONTROL_SOURCE = '438907f3a5ef681d6cde9012a66c7336fa642546'
OUT = ROOT / 'report'
HELPER = ARTIFACTS / 'antmaze_geodesic_gamma999_250k/report_results.py'
spec = importlib.util.spec_from_file_location('verified_geodesic_report', HELPER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
reward_path = WORKTREE / 'antmaze_experiments/progress_reward.py'
reward_bytes = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
                                       SOURCE + ':antmaze_experiments/progress_reward.py'])
reward = ModuleType('frozen_treatment_reward')
reward.__file__ = str(reward_path)
exec(compile(reward_bytes, str(reward_path), 'exec'), reward.__dict__)
upstream_bytes = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
    SOURCE + ':antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py'])
assert reward.UPSTREAM.read_bytes() == upstream_bytes
# This exact source preserves old profiles bit-for-bit (covered by source tests).
# Reusing endpoint/route validation does not relabel legacy reward values.
base.reward = reward
MODES = ('policy', 'native')
CONDITIONS = ('floor-exp(-5)', 'floor-0.5', 'floor-1')


def read(path):
    return json.loads(path.read_text())


def collect():
    runs = {}
    for campaign, source, treatment in (
            (ARTIFACTS/'antmaze_geodesic_gamma999_250k', CONTROL_SOURCE, False),
            (ROOT, SOURCE, True)):
        for host in (campaign/'results').glob('vast-heechan-*'):
            manifest = read(host/'manifest.json')
            assert manifest['source_commit'] == source
            for job in manifest['jobs']:
                if job['task'] != 'v4' or job['temperature'] != 1.:
                    continue
                name = f"floor-{job['teacher_std_floor']:g}" if treatment else 'floor-exp(-5)'
                assert name in CONDITIONS and name not in runs
                run = dict(job=job, source=source, evaluations={m: [] for m in MODES}, verified_final=False)
                runs[name] = run
                folder = host/'runs'/job['id']
                if not (folder/'config.json').exists():
                    continue
                cfg = read(folder/'config.json')
                expected = reward.NO_COST_PROFILE
                assert cfg['source_commit'] == source and cfg['seed'] == 0
                assert cfg['reward_profile'] == expected
                assert cfg['reward_specification'] == reward.specification('v4', expected)
                assert cfg['temperature'] == job['temperature'] and cfg['native']['alg']['gamma'] == .999
                assert cfg['native']['alg']['actor']['proposal_std'] == job.get('teacher_std_floor', 0.006737946999085467)
                assert cfg['native']['alg']['actor']['log_std_max'] == -1.
                assert cfg['native']['alg']['actor']['num_policy_samples'] == 64
                assert cfg['dacer_target_entropy_per_dim'] == .7 and cfg['dacer_interval_updates'] == 500
                assert cfg['dacer_enabled'] and not cfg['noveld_enabled'] and cfg['eval_starts'] == 'upstream'
                for mode in MODES:
                    for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                        if path.with_name('rollouts.npz').exists():
                            run['evaluations'][mode].append(base.evaluate(path, cfg, name, mode))
                if (folder/'result.json').exists():
                    proof = read(folder/'result.json')
                    assert proof['completed'] and proof['source_commit'] == source
                    assert proof['steps'] == 258304 and proof['updates'] == 7816
                    assert proof['checkpoint']['environment_reward_verified']
                    assert proof['checkpoint']['progress_replay_verified']
                    if treatment:
                        assert proof['teacher_std_floor'] == job['teacher_std_floor']
                        assert read(folder/'teacher-proposal-verification.json')['verified']
                    run['verified_final'] = True
    return runs


def main():
    OUT.mkdir(exist_ok=True)
    runs = collect()
    history = [e['row'] for run in runs.values() for es in run['evaluations'].values() for e in es]
    latest = [es[-1]['row'] for run in runs.values() for es in run['evaluations'].values() if es]
    matched = []
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
        by_condition = {c: {e['row']['step']: e for e in runs.get(c, {}).get('evaluations', {}).get(mode, [])}
                        for c in CONDITIONS}
        common = set.intersection(*(set(rows) for rows in by_condition.values()))
        step = max(common) if common else None
        for ax, condition in zip(axes, CONDITIONS):
            evaluation = by_condition[condition].get(step)
            base.draw(ax, 'v4', evaluation, condition)
            if evaluation:
                matched.append(evaluation['row'])
        fig.suptitle('v4 | teacher std-floor only | original geodesic reward / T1\n'
                     f'{mode}: identical full start, no external DACER noise, seed0; matched total step', fontsize=12)
        fig.savefig(OUT/f'matched_trajectories_{mode}.png', dpi=155)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout='constrained')
        for condition in CONDITIONS:
            rows = [e['row'] for e in runs.get(condition, {}).get('evaluations', {}).get(mode, [])]
            if not rows:
                continue
            x = [r['step']/1000 for r in rows]
            axes[0].plot(x, [100*r['success_rate'] for r in rows], marker='.', label=condition)
            axes[1].plot(x, [100*r['minority_success_rate'] for r in rows], marker='.', label=condition)
            axes[2].plot(x, [r['closest_euclidean_goal_distance_mean'] for r in rows], marker='.', label=condition)
        for ax, title in zip(axes, ('Any-goal success (%)', 'Less-used successful route (%)', 'Closest Euclidean goal distance (m)')):
            ax.set_title(title); ax.set_xlabel('Total transitions (k)'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'v4 | {mode} | failures stay in every denominator; one training seed', fontsize=11)
        fig.savefig(OUT/f'learning_curves_{mode}.png', dpi=155)
        plt.close(fig)
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), source_commit=SOURCE,
        control_source=CONTROL_SOURCE, goal_achieved=False,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        reward_implementation_sha256=hashlib.sha256(reward_bytes).hexdigest(),
        final_verified=[k for k, r in runs.items() if r['verified_final']],
        history=history, latest=latest, exact_step_matched=matched,
        caveats=['One training seed, different conditions are not seed replications.',
                 'Only existing teacher std floor changes; actor sigma, reward, gamma.999 and T1 are fixed.',
                 'Broader teacher candidates do not guarantee better learned Q or exploration.',
                 'Entering both sides alone is not successful route diversity or later retention.'])
    (OUT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# v4 teacher 후보 폭 비교', '',
        '기존 teacher floor exp(-5)와0.5·1.0을 비교합니다. 알고리즘 구조와 물리 환경은 동일합니다. '
        'teacher 표본과 logq에 같은 floor를 쓰며 actor sigma 범위[-5,-1],T1,discount.999,기존 geodesic 보상은 유지합니다.', '',
        '| 조건 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 목표별 성공 경로 | 성공률 |',
        '|---|---|---:|---:|---|---|---|---:|']
    for r in latest:
        lines.append(f"|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|{r['successful_goal_routes']}|{r['success_rate']:.1%}|")
    lines += ['', '![같은 예산 직접 정책](matched_trajectories_policy.png)',
        '![직접 정책 학습 추이](learning_curves_policy.png)', '',
        '궤적 그림은 세 조건 모두 저장된 정확히 같은 step을 비교합니다. 표는 최신 결과라 step이 다를 수 있습니다. '
        'v4 원래 고정 full state, seed0 하나입니다. 직접 정책은 random z와 conditional sigma를 포함하며 외부 DACER 잡음은 없습니다. '
        'mu-only는 별도 보조 결과입니다. 각 reward profile로 보상 합·시작 상태·성공 목표·원시 파일 SHA256을 검증했습니다. '
        '통로 진입만으로 두 성공 경로나 장기간 유지를 주장하지 않습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(OUT), 'rows': len(history), 'final_verified': payload['final_verified']}))


if __name__ == '__main__':
    main()
