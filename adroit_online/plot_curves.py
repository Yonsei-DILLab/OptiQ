"""Plot frozen evaluation records, by seed, at exact shared training steps.

This is read-only analysis. It never connects to learners or changes their files.
The input contains per-file hashes and episode-count-verified success metrics.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

TASKS = ('door', 'hammer', 'pen', 'relocate')
SEEDS = (4, 5)
METHODS = ('sac', 'td3', 'ibolt')
STYLES = {
    'sac': dict(color='#2171B5', linestyle='-', marker='o', label='SAC'),
    'td3': dict(color='#E07818', linestyle='--', marker='s', label='TD3'),
    'ibolt': dict(color='#19856B', linestyle='-.', marker='^', label='iBOLT (T = 0.25)'),
}
TOTAL = 3_000_064
INTERVAL = 50_176


def load_snapshot(path):
    snapshot = json.loads(path.read_text())
    runs = {}
    for host, data in snapshot.items():
        for name, row in data['runs'].items():
            s = row['spec']; key = (s['task'], s['seed'], s['method'])
            assert key not in runs, key
            assert s['profile'] == 'e256-u025' and s['num_envs'] == 256 and s['utd'] == .25
            assert s['seed'] in SEEDS and s['task'] in TASKS and s['method'] in METHODS
            assert s['steps'] == TOTAL and s['demonstration_ratio'] == 0
            assert not s['her'] and not s['behavior_cloning']
            assert not s['normalize_observations'] and not s['normalize_rewards']
            if s['method'] == 'ibolt': assert s['temperature'] == .25
            evaluations = {}
            for e in row['evaluations']:
                step = e['step']; assert step not in evaluations
                assert step == TOTAL or step % INTERVAL == 0
                expected = 'stochastic_z' if s['method'] == 'ibolt' else 'deterministic'
                assert e['primary'] == expected
                assert e['episodes'] == (1000 if step == TOTAL else 100)
                mode = e['modes'][expected]
                for metric in ('success_any', 'success_25_steps', 'success_final'):
                    assert 0 <= mode[metric] <= 1
                    assert abs(mode[metric] - mode[metric + '_count'] / e['episodes']) < 1e-12
                assert mode['success_25_steps'] <= mode['success_any']
                assert mode['success_final'] <= mode['success_any']
                evaluations[step] = e
            runs[key] = dict(name=name, host=host, spec=s, evaluations=evaluations)
    return snapshot, runs


def align(runs):
    panels = {}
    for task in TASKS:
        for seed in SEEDS:
            available = [m for m in METHODS if (task, seed, m) in runs
                         and runs[task, seed, m]['evaluations']]
            assert 'sac' in available and 'td3' in available
            sets = [set(runs[task, seed, m]['evaluations']) for m in available]
            steps = sorted(set.intersection(*sets)); assert steps and steps[0] == 0
            for step in steps:
                assert len({runs[task, seed, m]['evaluations'][step]['episodes'] for m in available}) == 1
            panels[task, seed] = dict(methods=available, steps=steps,
                                     missing=[m for m in METHODS if m not in available])
    return panels


def plot(runs, panels, metric, title, output, stamp, matched=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.titlesize': 15, 'axes.labelsize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.80, hspace=.37, wspace=.15)
    fig.suptitle(title, x=.065, y=.978, ha='left', fontsize=22, fontweight='bold', color='#182A39')
    policy = 'Exact shared evaluation steps in each panel' if matched else 'All recorded evaluation steps'
    fig.text(.065, .928, f'Adroit Sparse-v2  |  256 environments  |  UTD 0.25  |  {policy}',
             fontsize=13, color='#526575')
    handles = [Line2D([0], [0], linewidth=2.2, markersize=5, **STYLES[m]) for m in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .897),
               ncol=3, frameon=False, fontsize=14, handlelength=3, columnspacing=3)
    for row, seed in enumerate(SEEDS):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]; panel = panels[task, seed]
            cutoff = panel['steps'][-1] / 1e6
            if matched and cutoff < TOTAL / 1e6:
                ax.axvspan(cutoff, TOTAL / 1e6, color='#E6EAEE', alpha=.70, linewidth=0, zorder=0)
                ax.axvline(cutoff, color='#8493A0', linestyle=':', linewidth=1, zorder=1)
            for index, method in enumerate(METHODS):
                if method not in panel['methods']: continue
                run = runs[task, seed, method]
                steps = panel['steps'] if matched else sorted(run['evaluations'])
                values = [100 * run['evaluations'][s]['modes'][run['evaluations'][s]['primary']][metric]
                          for s in steps]
                ax.plot(np.asarray(steps) / 1e6, values, linewidth=2.0, markersize=3.4,
                        markevery=(index, 5), alpha=.97, zorder=3 + index, **STYLES[method])
            ax.set_title(task.title(), pad=9, fontweight='semibold')
            ax.set_xlim(0, TOTAL / 1e6 + .035); ax.set_ylim(-3, 104)
            ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis='y', color='#DCE2E7', linewidth=.65, zorder=0)
            ax.tick_params(axis='both', length=3, color='#93A2AD')
            ax.spines['left'].set_color('#A8B3BC'); ax.spines['bottom'].set_color('#A8B3BC')
            ax.set_facecolor('#FCFDFE')
            if col == 0: ax.set_ylabel(f'Seed {seed}\nSuccess rate (%)', labelpad=10)
            if row == 1: ax.set_xlabel('Training transitions (millions)', labelpad=7)
            label = f'Matched through {cutoff:.3f}M' if matched else 'No extrapolation'
            if panel['missing']: label = 'iBOLT queued\nSAC / TD3 only'
            ax.text(.035, .95, label, transform=ax.transAxes, ha='left', va='top',
                    fontsize=10.5, color='#566875',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.90, pad=3))
    definition = ('Episode success = native success at least once during the 200-step episode.'
                  if metric == 'success_any' else
                  'Episode success = at least 25 successful steps out of 200; steps need not be consecutive.')
    fig.text(.065, .122, definition, fontsize=11.5, color='#334C5D')
    fig.text(.065, .088,
             'Policies: SAC / TD3 deterministic; iBOLT fresh latent z + conditional mean. No external evaluation noise.',
             fontsize=11, color='#526575')
    fig.text(.065, .056,
             '100 episodes per regular evaluation; 1,000 at 3M. No smoothing or seed averaging. '
             + ('Gray area: not compared yet.' if matched else 'Each line stops at its latest recorded evaluation.'),
             fontsize=10.5, color='#526575')
    fig.text(.065, .024, f'Snapshot: {stamp}', fontsize=10.5, color='#71828E')
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=180, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--analysis-commit', required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    snapshot, runs = load_snapshot(a.snapshot); panels = align(runs)
    latest = max(d['observed_unix'] for d in snapshot.values())
    stamp = datetime.fromtimestamp(latest, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    for metric, stem, title in (
        ('success_any', 'instant_success', 'Any-step success'),
        ('success_25_steps', 'success_25_steps', 'At least 25 successful steps'),
    ):
        plot(runs, panels, metric, title, a.output / (stem + '_matched'), stamp)
        plot(runs, panels, metric, title, a.output / (stem + '_full_progress'), stamp, matched=False)
    audit = dict(snapshot_kst=stamp, snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
                 analysis_commit=a.analysis_commit,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 alignment='Exact step intersection among available algorithms, independently per task and seed; no interpolation of observations',
                 smoothing=False, seed_averaging=False, panels=[])
    for (task, seed), panel in panels.items():
        last = panel['steps'][-1]
        endpoints = {}
        for m in panel['methods']:
            r = runs[task, seed, m]; e = r['evaluations'][last]
            endpoints[m] = dict(run=r['name'], source_commit=r['spec']['source_commit'],
                                episodes=e['episodes'], primary=e['primary'],
                                **e['modes'][e['primary']])
        audit['panels'].append(dict(task=task, seed=seed, **panel, endpoints=endpoints))
    (a.output / 'alignment_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(snapshot=stamp, runs=len(runs), panels=len(panels),
                         three_method_panels=sum(len(x['methods']) == 3 for x in panels.values()),
                         output=str(a.output)), indent=2))


if __name__ == '__main__':
    main()
