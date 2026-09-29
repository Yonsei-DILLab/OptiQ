"""Read-only, condition-aware route retention report for the T3 follow-up."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent
HELPER = ARTIFACTS/'antmaze_horizon_temperature_250k/report_results.py'
spec = importlib.util.spec_from_file_location('retention_validation', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
OUT = ROOT/'report'
SERIES = {
    'T1_1M': dict(root=ARTIFACTS/'antmaze_gamma999_retention_1m',
        job='v3-optiq-gamma999-H0.7-i500-1m-s0', temperature=1.,
        source='d2662636c4b8253efed8d9d0fc2c1af6a5b1da15', budget=1008384,
        label='T1, completed 1M control', color='#7e8992'),
    'T3_250k': dict(root=ARTIFACTS/'antmaze_horizon_temperature_250k',
        job='v3-optiq-gamma999_temp3-H0.7-i500-250k-s0', temperature=3.,
        source='eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45', budget=258304,
        label='T3, completed 250k screen', color='#d48a34'),
    'T3_1M': dict(root=ROOT, job='v3-optiq-gamma999-T3-H0.7-i500-1m-s0',
        temperature=3., source='8eb830d553f93e1360bcee5f6c9649e6392e5d25', budget=1008384,
        label='T3, fresh 1M follow-up', color='#277cb5')}


def collect():
    series = {}
    final_proofs = {}
    for name, item in SERIES.items():
        folder = item['root']/'results/vast-heechan-180/runs'/item['job']
        series[name] = {mode: [] for mode in ('policy', 'native')}
        if not (folder/'config.json').exists():
            assert name == 'T3_1M', ('Missing completed control', folder)
            continue
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['source_commit'] == item['source'] and cfg['seed'] == 0
        assert cfg['temperature'] == item['temperature']
        assert cfg['native']['alg']['gamma'] == .999
        assert cfg['dacer_target_entropy_per_dim'] == .7 and cfg['dacer_interval_updates'] == 500
        assert cfg['eval_starts'] == 'upstream' and not cfg['noveld_enabled']
        assert cfg['reward_specification']['formula'] == '100*(d(current)-d(next))'
        if (folder/'result.json').exists():
            proof = json.loads((folder/'result.json').read_text())
            assert proof['completed'] and proof['source_commit'] == item['source']
            assert proof['steps'] == item['budget']
            assert proof['updates'] == (item['budget']-8192)//256*8
            assert proof['checkpoint']['environment_reward_verified'] and proof['checkpoint']['readback_verified']
            final_proofs[name] = proof['checkpoint']
        for mode in ('policy', 'native'):
            for p in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                if not p.with_name('rollouts.npz').exists():
                    continue
                e = helper.evaluate_saved(p, cfg, name, mode)
                e['row']['series'] = name
                series[name][mode].append(e)
    return series, final_proofs


def main():
    OUT.mkdir(exist_ok=True)
    series, final_proofs = collect()
    stop_path = ROOT/'results/vast-heechan-180/screen-stop.json'
    screen_stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    if screen_stop is not None:
        assert screen_stop['decision'] == 'stopped_early_route_loss'
        assert screen_stop['process_group_terminated'] and not screen_stop['training_complete']
        SERIES['T3_1M']['label'] = 'T3, stopped early after route loss'
    latest = [es[-1]['row'] for modes in series.values() for es in modes.values() if es]
    history = [e['row'] for modes in series.values() for es in modes.values() for e in es]
    prefix = []
    for mode in ('policy', 'native'):
        short = {e['row']['step']: e['row'] for e in series['T3_250k'][mode]}
        for e in series['T3_1M'][mode]:
            r = e['row']
            if r['step'] not in short:
                continue
            with np.load(r['raw_path'], allow_pickle=False) as a, np.load(short[r['step']]['raw_path'], allow_pickle=False) as b:
                equal = {key: bool(a[key].shape == b[key].shape and np.array_equal(a[key], b[key], equal_nan=True))
                         for key in ('xy', 'goals', 'returns', 'initial_full_state')}
            assert all(equal.values()), (mode, r['step'], equal)
            prefix.append(dict(mode=mode, step=r['step'], exact_equal=equal))
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), latest=latest, history=history,
        final_checkpoint_proofs=final_proofs, same_seed_prefix=prefix,
        screen_stop=screen_stop,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_validator_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        limitations=['All training seed0; fresh longer run is not a resume or independent replication.',
            'policy includes conditional sigma; native uses random-z mu-only.',
            'Original fixed complete start state, no external DACER evaluation noise.',
            'Corridor entry is not successful route retention; compare matched steps.'])
    (OUT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    for mode in ('policy', 'native'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        keys = ('success_rate', 'minority_entry_rate', 'minority_success_rate', 'mean_closest_goal_distance')
        titles = ('Goal success / all episodes', 'Minority corridor entry / all episodes',
                  'Minority successful route / all episodes', 'Closest goal distance, mean (m)')
        for ax, key, title in zip(axes.flat, keys, titles):
            for name, modes in series.items():
                rows = [e['row'] for e in modes[mode]]
                ax.plot([r['step']/1000 for r in rows], [r[key] for r in rows],
                    color=SERIES[name]['color'], marker='o', ms=3, label=SERIES[name]['label'])
            if key != 'mean_closest_goal_distance':
                ax.set_ylim(-.02, 1.02 if key == 'success_rate' else .52)
            ax.set_title(title); ax.set_xlabel('Total transitions (k)'); ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'v3 gamma .999: T1 vs T3 successful-route retention | {mode}\nSame training seed; different budgets, no pooled samples or seed error bars.')
        fig.savefig(OUT/f'learning_curves_{mode}.png', dpi=160); plt.close(fig)
    walls, goals, bounds = helper.maze_geometry('v3')
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), layout='constrained')
    by_mode = {mode: {e['row']['step']: e for e in series['T3_1M'][mode]} for mode in ('policy', 'native')}
    common = sorted(by_mode['policy'].keys() & by_mode['native'].keys())
    step = common[-1] if common else None
    for ax, mode in zip(axes, ('policy', 'native')):
        for x0, y0, x1, y1 in walls:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor='#e2e6e9', edgecolor='#bdc4cb', lw=.4))
        if step is not None:
            e = by_mode[mode][step]; r = e['row']
            for i in np.argsort(e['goals']>0):
                p = e['points'][i]; ok = e['goals'][i]>0
                ax.plot(p[:, 0], p[:, 1], color=helper.ROUTE_COLORS[e['labels'][i]],
                    alpha=.5 if ok else .2, lw=1 if ok else .7)
            entries = ' '.join(f'{helper.SHORT[k]}:{v}' for k, v in r['route_counts'].items())
            successes = ' '.join(f'{helper.SHORT[k]}:{v}' for k, v in r['successful_route_counts'].items()) or 'none'
            ax.set_title(f"{mode} | {r['step']/1000:.1f}k | n={r['episodes']}\nentry {entries}\nsuccessful routes {successes}", fontsize=10)
        else:
            ax.set_title(mode+' | no saved raw evaluation')
        ax.scatter(*goals.T, marker='*', s=100, c='#38a35f', edgecolor='white', lw=.5)
        ax.scatter(0, 0, c='black', s=12)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect('equal')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    phase = 'stopped early; planned1M incomplete' if screen_stop else 'fresh 1M-budget follow-up'
    fig.suptitle(f'v3 gamma .999, T3, H/d+.7 | {phase}\nLatest matched fixed-start checkpoint; no external DACER noise.', fontsize=11)
    fig.savefig(OUT/'latest_trajectories.png', dpi=160); plt.close(fig)
    lines = ['# v3 gamma .999·T3 경로 유지 추적', '',
        '완료한 T1 대조군, T3 250k 탐색 실험, 별도의 fresh T3 1M 실험을 비교합니다. 모두 seed0이며 짧은 실험과 긴 실험을 독립 seed로 세지 않습니다.', '',
        '| 실험 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |',
        '|---|---|---:|---:|---|---|---:|']
    for r in latest:
        lines.append(f"|{r['series']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|{r['success_rate']:.1%}|")
    lines += ['', '![현재 궤적](latest_trajectories.png)', '![학습 추이](learning_curves_policy.png)', '',
        '원시 궤적의 동일 full state·성공 위치·보상·SHA256 및 완료한 실험의 최종 체크포인트 증명을 검증합니다. '
        '학습 알고리즘은 바꾸지 않았으며 통로 진입과 그 통로를 통한 목표 성공을 분리해 기록합니다.']
    if screen_stop:
        lines += ['', f"T3 후속은450k·500k 평가에서 양쪽 모드 모두 반대 경로를 잃어 조기 중단했습니다. 마지막 기록된 학습 step은{screen_stop['last_logged_progress']['step']}입니다. "
                  '계획한1M을 완료한 실험이 아니며 최종100회 평가·전체 replay 체크포인트는 없습니다. 중간 체크포인트, 원시 평가, 로그와 frozen source는 보존했습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(latest=latest, final_verified=list(final_proofs), prefix_checks=len(prefix)), indent=2))


if __name__ == '__main__':
    main()
