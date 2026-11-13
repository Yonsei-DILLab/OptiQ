"""Compare verified old baselines and new iBOLT, with exact step/reset pairing."""
import argparse
import hashlib
import json
from pathlib import Path

from .plot_curves import METHODS, STYLES


def matched_runs(baselines, snapshot):
    runs = {}
    for name, row in baselines['runs'].items():
        s = row['spec']
        if s['method'] not in ('sac', 'td3') or s['task'] not in ('door', 'hammer') or s['profile'] != 'e64-u1':
            continue
        converted = []
        for e in row['evaluations']:
            if e['kind'] == 'uniform-random':
                continue
            converted.append(dict(e, reset_seeds_sha256=e['reset_sha256']))
        runs[name] = dict(spec=s, evaluations=converted)
    assert len(runs) == 8
    assert not (runs.keys() & snapshot['runs'].keys())
    runs.update(snapshot['runs'])
    index = {}
    for name, row in runs.items():
        s = row['spec']
        assert s['profile'] == 'e64-u1' and s['num_envs'] == 64 and s['utd'] == 1
        assert s['steps'] == 3000064 and s['expected_updates'] == 2989824
        assert s['task'] in ('door', 'hammer') and s['seed'] in (4, 5)
        assert s['demonstration_ratio'] == 0 and not s['behavior_cloning'] and not s['her']
        variant = f"ibolt_t{round(s['temperature'] * 100):03d}" if s['method'] == 'ibolt' else s['method']
        assert variant in METHODS
        key = (s['task'], s['seed'], variant); assert key not in index
        evaluations = {}
        for e in row['evaluations']:
            step = e['step']; assert step not in evaluations
            assert step == 3000064 or step % 50176 == 0
            assert e['episodes'] == (1000 if step == 3000064 else 100)
            assert e['primary'] == ('stochastic_z' if s['method'] == 'ibolt' else 'deterministic')
            for metric in ('success_any', 'success_25_steps'):
                m = e['modes'][e['primary']]
                assert abs(m[metric] - m[metric + '_count'] / e['episodes']) < 1e-12
            evaluations[step] = e
        index[key] = dict(name=name, spec=s, evaluations=evaluations)
    for task in ('door', 'hammer'):
        for seed in (4, 5):
            groups = [r['evaluations'] for (t, s, _), r in index.items() if (t, s) == (task, seed)]
            for step in set.union(*(set(g) for g in groups)):
                assert len({g[step]['reset_seeds_sha256'] for g in groups if step in g}) == 1
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    b = json.loads(a.baselines.read_text()); snapshot = json.loads(a.snapshot.read_text())
    runs = matched_runs(b, snapshot)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    for metric, title in [('success_any', 'Native success at least once'),
                          ('success_25_steps', 'At least 25 successful steps (cumulative)')]:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharex=True, sharey=True)
        fig.subplots_adjust(top=.80, bottom=.17, hspace=.30, wspace=.15)
        fig.suptitle('Adroit Sparse-v2: env64 / UTD1\n' + title, y=.98, fontsize=18)
        fig.legend(handles=[Line2D([], [], **STYLES[m]) for m in METHODS],
                   loc='upper center', bbox_to_anchor=(.5, .90), ncol=5, frameon=False, fontsize=10)
        for i, seed in enumerate((4, 5)):
            for j, task in enumerate(('door', 'hammer')):
                ax = axes[i, j]; pending = []
                for variant in METHODS:
                    r = runs.get((task, seed, variant))
                    if not r or not r['evaluations']:
                        pending.append(STYLES[variant]['label']); continue
                    steps = sorted(r['evaluations'])
                    y = [100 * r['evaluations'][s]['modes'][r['evaluations'][s]['primary']][metric] for s in steps]
                    ax.plot([s / 1e6 for s in steps], y, linewidth=1.8, markersize=2.8,
                            markevery=5, **STYLES[variant])
                ax.set_title(f'{task.title()} | seed {seed}')
                ax.set_xlim(0, 3.02); ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
                ax.grid(alpha=.20)
                if j == 0: ax.set_ylabel('Success rate (%)')
                if i == 1: ax.set_xlabel('Training transitions (millions)')
                if pending: ax.text(.03, .97, 'Pending: ' + ', '.join(pending), transform=ax.transAxes,
                                    va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        fig.text(.12, .09, '100 episodes / regular evaluation; 1,000 / final. Same reset seeds and exact step grid.', fontsize=10)
        fig.text(.12, .06, 'SAC/TD3: deterministic. iBOLT: fresh latent z + conditional mean. No smoothing.', fontsize=10)
        fig.text(.12, .03, 'Partial curves stop at their actual last evaluation; full baselines are retained for reference.', fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(a.output / f'{metric}.{ext}', dpi=180, facecolor='white')
        plt.close(fig)
    audit = dict(baselines_sha256=hashlib.sha256(a.baselines.read_bytes()).hexdigest(),
                 snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
                 observed_unix=snapshot['observed_unix'], queue=snapshot['queue'], latest=[])
    for key, row in sorted(runs.items()):
        if row['evaluations']:
            step = max(row['evaluations']); e = row['evaluations'][step]
            audit['latest'].append(dict(run_id=row['name'], source_commit=row['spec']['source_commit'],
                                       step=step, episodes=e['episodes'], primary=e['primary'],
                                       metrics=e['modes'][e['primary']]))
    (a.output / 'comparison.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(runs=len(runs), output=str(a.output))))


if __name__ == '__main__':
    main()
