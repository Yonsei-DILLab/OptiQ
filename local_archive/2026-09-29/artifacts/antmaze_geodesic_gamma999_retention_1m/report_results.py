"""Validate and plot the existing geodesic candidates without mutating training."""
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
SHORT = ROOT.parent/'antmaze_geodesic_gamma999_250k'
HELPER = SHORT/'report_results.py'
spec = importlib.util.spec_from_file_location('geodesic_validation', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
OUT = ROOT/'report'
SOURCES = {'short': '438907f3a5ef681d6cde9012a66c7336fa642546',
           'long': '8e9d7d3c2c79f797654ccfb21913d2c000b89f71'}
TASKS = ('v1', 'v4')
MODES = ('policy', 'native')


def collect():
    data = {}
    proofs = {}
    for name, root, budget in (('short', SHORT, 258304), ('long', ROOT, 1008384)):
        host = root/'results/vast-heechan-180'
        if not (host/'manifest.json').exists():
            assert name == 'long'
            continue
        manifest = json.loads((host/'manifest.json').read_text())
        assert manifest['source_commit'] == SOURCES[name]
        for job in manifest['jobs']:
            task = job['task']
            if task not in TASKS:
                continue
            folder = host/'runs'/job['id']
            data[task, name] = {mode: [] for mode in MODES}
            if not (folder/'config.json').exists():
                assert name == 'long'
                continue
            cfg = json.loads((folder/'config.json').read_text())
            assert cfg['source_commit'] == SOURCES[name] and cfg['seed'] == 0
            assert cfg['temperature'] == 1 and cfg['native']['alg']['gamma'] == .999
            assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
            assert cfg['dacer_target_entropy_per_dim'] == .7 and cfg['dacer_interval_updates'] == 500
            assert cfg['eval_starts'] == 'upstream' and cfg['reward_profile'] == helper.reward.NO_COST_PROFILE
            reset = 'natural' if task == 'v1' else 'fixed'
            for mode in MODES:
                for path in sorted((folder/'evaluations').glob(f'*/{mode}-{reset}/summary.json')):
                    if path.with_name('rollouts.npz').exists():
                        data[task, name][mode].append(helper.evaluate(path, cfg, name, mode))
            if (folder/'result.json').exists():
                result = json.loads((folder/'result.json').read_text())
                assert result['completed'] and result['source_commit'] == SOURCES[name]
                assert result['steps'] == budget and result['updates'] == (budget-8192)//256*8
                assert result['checkpoint']['readback_verified'] and result['checkpoint']['environment_reward_verified']
                proofs[f'{task}/{name}'] = result['checkpoint']
    return data, proofs


def main():
    OUT.mkdir(exist_ok=True)
    data, proofs = collect()
    stop_paths = (ROOT/'results/vast-heechan-180/screen-stop.json', ROOT/'screen-stop.json')
    stop = next((json.loads(p.read_text()) for p in stop_paths if p.exists()), None)
    if stop:
        assert stop['source_commit'] == SOURCES['long'] and not stop['planned_budget_completed']
        assert stop['stopped_job'].startswith('v4-')
        assert 'v4/long' not in proofs
    latest = [es[-1]['row'] for modes in data.values() for es in modes.values() if es]
    history = [e['row'] for modes in data.values() for es in modes.values() for e in es]
    prefix = []
    for task in TASKS:
        for mode in MODES:
            previous = {e['row']['step']: e['row'] for e in data.get((task, 'short'), {}).get(mode, [])}
            for e in data.get((task, 'long'), {}).get(mode, []):
                r = e['row']
                if r['step'] not in previous:
                    continue
                with np.load(r['raw_path'], allow_pickle=False) as a, np.load(previous[r['step']]['raw_path'], allow_pickle=False) as b:
                    equal = {k: bool(a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=True))
                             for k in ('xy', 'goals', 'returns', 'initial_full_state')}
                assert all(equal.values()), (task, mode, r['step'], equal)
                prefix.append(dict(task=task, mode=mode, step=r['step'], exact_equal=equal))
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), latest=latest, history=history,
        final_checkpoint_proofs=proofs, same_seed_prefix=prefix, screen_stop=stop,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reward_raw_validator_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        limitations=['All training seed0; longer runs are fresh, not resumes or independent seeds.',
            'v1 uses native random starts; v4 original identical full state.',
            'policy includes conditional sigma; native is random-z mu-only; no external evaluation noise.',
            'Successful-route retention and corridor entry must be distinguished.',
            'Geodesic is the existing XY point-agent potential without physical body inflation.'])
    (OUT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    plt.rcParams.update({'font.size': 9})
    for mode in MODES:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
        for row, task in enumerate(TASKS):
            for col, (key, label) in enumerate((('success_rate', 'Goal success'),
                    ('minority_success_rate', 'Minority successful route / all'),
                    ('minority_entry_rate', 'Minority corridor entry / all'))):
                ax = axes[row, col]
                for name, color in (('short', '#d48a34'), ('long', '#277cb5')):
                    rows = [e['row'] for e in data.get((task, name), {}).get(mode, [])]
                    label_name = name+' (stopped early)' if stop and task=='v4' and name=='long' else name
                    ax.plot([r['step']/1000 for r in rows], [r[key] for r in rows],
                            label=label_name, color=color, marker='o', ms=3)
                ax.set_title(f'{task} | {label}'); ax.set_xlabel('Total transitions (k)')
                ax.set_ylim(-.02, 1.02 if col == 0 else .52); ax.grid(alpha=.2)
        axes[0, 0].legend()
        axes[1, 0].legend()
        fig.suptitle(f'Geodesic gamma .999 / T1 / H/d+.7 | {mode}\nSame training seed, no pooled rollouts; matched short prefix and fresh longer runs.')
        fig.savefig(OUT/f'learning_curves_{mode}.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), layout='constrained')
    for row, task in enumerate(TASKS):
        by_mode = {mode: {e['row']['step']: e for e in data.get((task, 'long'), {}).get(mode, [])}
                   for mode in MODES}
        common = sorted(by_mode['policy'].keys() & by_mode['native'].keys())
        step = common[-1] if common else None
        for col, mode in enumerate(MODES):
            label = 'stopped early' if stop and task=='v4' else 'long run'
            helper.draw(axes[row, col], task, by_mode[mode].get(step), f'{task} | {label} | {mode}')
    fig.suptitle('Latest matched checkpoints of 1M-budget geodesic candidates\nv1 native random starts; v4 original fixed full state. No external DACER noise.', fontsize=12)
    fig.savefig(OUT/'latest_trajectories.png', dpi=160); plt.close(fig)
    lines = ['# Geodesic v1/v4 양방향 성공 유지 검증', '',
        '250k 탐색 실험과 동일한 설정으로 새로 시작한1M 실험을 비교합니다. 모두 seed0이고 독립 학습 seed 또는 체크포인트 재개가 아닙니다.', '',
        '| 환경 | 예산 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |',
        '|---|---|---|---:|---:|---|---|---:|']
    if stop:
        lines[3:3] = [f"v4는 반대쪽 성공 경로 소실로 조기 중단했습니다. 마지막 학습 로그는 {stop['last_logged_step']}total step이며, 1M 완료·최종100회 평가·최종 full checkpoint가 아닙니다. v1의 완료 결과는 보존했습니다.", '']
    for r in latest:
        lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|{r['success_rate']:.1%}|")
    lines += ['', '![현재 궤적](latest_trajectories.png)', '![직접 정책 추이](learning_curves_policy.png)', '',
        'v1은 원래 랜덤 시작이므로 양방향 성공만으로 동일 상태의 다중 경로를 입증하지 않습니다. '
        'v4는 원래 고정 full state입니다. 그림은 환경마다 두 평가 모드의 최신 공통 checkpoint를 사용합니다. 표는 모드별 최신 저장 평가라 step이 다를 수 있습니다. '
        '직접 정책과 mu-only를 합치지 않고, 실패도 모두 분모에 포함합니다. '
        '원시 보상·goal·시작 상태·SHA256 및 완료한 실험의 체크포인트 증명을 검증했습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(latest=latest, final_verified=list(proofs), prefix_checks=len(prefix)), indent=2))


if __name__ == '__main__':
    main()
