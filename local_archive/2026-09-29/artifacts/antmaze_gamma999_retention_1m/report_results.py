"""Validate existing v3 trajectories and track successful-route retention."""
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
HELPER = ARTIFACTS / 'antmaze_horizon_temperature_250k/report_results.py'
spec = importlib.util.spec_from_file_location('route_validation', HELPER)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
OUT = ROOT / 'report'
SERIES = {
    'control': dict(root=ARTIFACTS/'antmaze_dacer_positive_500k',
                    job='v3-optiq-H0.7-i500-500k-s0',
                    source='2564b59faa0d319eece496b93eff0f19359efc37', gamma=.99,
                    label='gamma .99 | 500k control, seed0', color='#818a92'),
    'short': dict(root=ARTIFACTS/'antmaze_horizon_temperature_250k',
                  job='v3-optiq-gamma999-H0.7-i500-250k-s0',
                  source='eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45', gamma=.999,
                  label='gamma .999 | 250k screen, seed0', color='#d48a34'),
    'long': dict(root=ROOT, job='v3-optiq-gamma999-H0.7-i500-1m-s0',
                 source='d2662636c4b8253efed8d9d0fc2c1af6a5b1da15', gamma=.999,
                 label='gamma .999 | fresh 1M follow-up, seed0', color='#277cb5')}


def load_series():
    result = {}
    for name, item in SERIES.items():
        folder = item['root']/'results/vast-heechan-180/runs'/item['job']
        if not (folder/'config.json').exists():
            result[name] = {mode: [] for mode in ('policy', 'native')}
            continue
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['source_commit'] == item['source'] and cfg['seed'] == 0
        assert cfg['native']['alg']['gamma'] == item['gamma']
        assert cfg['temperature'] == 1 and cfg['dacer_target_entropy_per_dim'] == .7
        assert cfg['dacer_interval_updates'] == 500 and cfg['eval_starts'] == 'upstream'
        assert cfg['reward_specification']['formula'] == '100*(d(current)-d(next))'
        modes = {}
        for mode in ('policy', 'native'):
            modes[mode] = []
            for p in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                if p.with_name('rollouts.npz').exists():
                    evaluation = validation.evaluate_saved(p, cfg, name, mode)
                    evaluation['row']['series'] = name
                    modes[mode].append(evaluation)
        result[name] = modes
    return result


def same_seed_prefix(series):
    checks = []
    for mode in ('policy', 'native'):
        short = {e['row']['step']: e['row'] for e in series['short'][mode]}
        for evaluation in series['long'][mode]:
            row = evaluation['row']
            if row['step'] not in short:
                continue
            previous = short[row['step']]
            with np.load(row['raw_path'], allow_pickle=False) as a, np.load(previous['raw_path'], allow_pickle=False) as b:
                same = {key: bool(a[key].shape == b[key].shape and np.array_equal(a[key], b[key], equal_nan=True))
                        for key in ('xy', 'goals', 'returns', 'initial_full_state')}
            checks.append(dict(mode=mode, step=row['step'], exact_equal=same,
                               independent_training_seed=False))
    return checks


def main():
    OUT.mkdir(exist_ok=True)
    series = load_series()
    records = [e['row'] for modes in series.values() for evaluations in modes.values() for e in evaluations]
    latest = [es[-1]['row'] for modes in series.values() for es in modes.values() if es]
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), goal_achieved=False,
                   report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   raw_validator_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                   latest=latest, history=records, same_seed_prefix=same_seed_prefix(series),
                   criteria='Goal completion requires successful distinct routes retained at later checkpoints and additional validation. Entry diversity is insufficient.',
                   limitations=['All runs use seed0; short and long runs are not independent seeds.',
                                'Long run starts fresh; it is not a checkpoint resume.',
                                'policy includes conditional sigma; native is random-z mu-only.',
                                'Original v3 fixed full state; external DACER noise excluded.',
                                'Different budgets; compare shared checkpoints before comparing final performance.'])
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
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'v3: goal-reaching and route retention | {mode}\nAll seed0, separate runs; no pooled episodes or seed-error bars.')
        fig.savefig(OUT/f'learning_curves_{mode}.png', dpi=160); plt.close(fig)
    walls, goals, bounds = validation.maze_geometry('v3')
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), layout='constrained')
    for ax, mode in zip(axes, ('policy', 'native')):
        for x0, y0, x1, y1 in walls:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e2e6e9',edgecolor='#bdc4cb',lw=.4))
        es = series['long'][mode]
        if es:
            e=es[-1];r=e['row']
            for i in np.argsort(e['goals']>0):
                p=e['points'][i];success=e['goals'][i]>0
                ax.plot(p[:,0],p[:,1],color=validation.ROUTE_COLORS[e['labels'][i]],
                        alpha=.5 if success else .2,lw=1 if success else .7)
            entry=' '.join(f'{validation.SHORT[k]}:{v}' for k,v in r['route_counts'].items())
            successes=' '.join(f'{validation.SHORT[k]}:{v}' for k,v in r['successful_route_counts'].items()) or 'none'
            ax.set_title(f"{mode} | {r['step']/1000:.1f}k | n={r['episodes']}\nentry {entry}\nsuccessful routes {successes}",fontsize=10,pad=10)
        else:
            ax.set_title(mode+' | no saved evaluation yet')
        ax.scatter(*goals.T,marker='*',s=100,c='#38a35f',edgecolor='white',lw=.5)
        ax.scatter(0,0,c='black',s=12)
        ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Fresh 1M-budget follow-up | latest v3 fixed-start trajectories\nPolicy: random z + conditional sigma. Native: random-z mu-only. No external DACER noise.',fontsize=11)
    fig.savefig(OUT/'latest_trajectories.png',dpi=160);plt.close(fig)
    lines=['# v3 gamma .999 장기 유지 검증','',
           '짧은 실험에서 남은 양방향 진입이 성공까지 이어지는지 확인하는 별도 fresh seed0 실험입니다. '
           '체크포인트 재개 또는 독립 seed로 해석하지 않습니다.','',
           '| 실험 | 평가 | total step | 횟수 | 첫 통로 | 성공 통로 | 성공률 |',
           '|---|---|---:|---:|---|---|---:|']
    for r in sorted(latest,key=lambda x:(x['series'],x['mode'])):
        lines.append(f"| {r['series']} | {r['mode']} | {r['step']} | {r['episodes']} | {r['route_counts']} | {r['successful_route_counts']} | {r['success_rate']:.1%} |")
    lines+=['','![현재 궤적](latest_trajectories.png)','![직접 정책 추이](learning_curves_policy.png)',
            '![mu-only 보조 추이](learning_curves_native.png)','',
            '동일한 원래 시작 full state에서 평가합니다. 외부 DACER 행동잡음은 제외합니다. 모든 원시 궤적은 시작상태·성공/goal endpoint·reward telescope·SHA256을 검증합니다. '
            '500k/750k/1M에서 양쪽 성공 경로가 유지되는지를 보고, 유망할 경우 추가 독립 평가·학습 seed 검증이 필요합니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT),latest=[r for r in latest if r['series']=='long'],
                          prefix_checks=len(payload['same_seed_prefix'])),indent=2))


if __name__ == '__main__':
    main()
