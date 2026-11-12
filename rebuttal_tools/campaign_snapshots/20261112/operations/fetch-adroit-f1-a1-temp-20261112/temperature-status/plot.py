"""Plot observed pilot temperatures on exact paired evaluation grids."""
from pathlib import Path
import json, datetime, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
d = json.loads((P / 'evidence.json').read_text())
temps = [.05, .1, .25, .5, 1.]
tasks = ['door', 'hammer', 'pen', 'relocate']
colors = dict(zip(temps, ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#7A6E00']))
rows = [r for r in d['rows'] if r['evaluations']]
maps = {}
for r in rows:
    c, j = r['config'], r['job']
    assert c['actor_hidden'] == c['critic_hidden'] == [256, 256]
    assert c['critic_layer_norm'] and c['dacer_enabled']
    assert c['offline_pretraining_updates'] == 0 and c['horizon'] == 200
    assert c['temperature'] == j['temperature']
    maps[j['id']] = {e['steps']: e for e in r['evaluations']
                     if len(e['zero_z']['episodes']) == 10}
    assert all(e['primary_mode'] == 'zero_z' for e in maps[j['id']].values())
common = sorted(set.intersection(*(set(m) for m in maps.values())))
global_step = max(common)
table, panels = [], []
panel_ends = []
for task in tasks:
    for seed in [5, 6]:
        rr = [r for r in rows if r['job']['task'] == task and r['job']['seed'] == seed]
        panel_ends.append(max(set.intersection(*(set(maps[r['job']['id']]) for r in rr))))
xmax = max(.55, math.ceil(max(panel_ends) / 250000) * .25)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(4, 2, figsize=(12.5, 13), sharey=True)
for i, task in enumerate(tasks):
    assert len({r['config']['demonstration_dataset_sha256'] for r in rows
                if r['job']['task'] == task}) == 1
    for k, seed in enumerate([5, 6]):
        ax = axes[i, k]
        rr = sorted([r for r in rows if r['job']['task'] == task and r['job']['seed'] == seed],
                    key=lambda r: r['job']['temperature'])
        grid = sorted(set.intersection(*(set(maps[r['job']['id']]) for r in rr)))
        for step in grid:
            es = [maps[r['job']['id']][step] for r in rr]
            for e in es:
                assert e['updates'] == es[0]['updates']
                assert e['env_seeds'] == es[0]['env_seeds']
                assert e['policy_seeds'] == es[0]['policy_seeds']
        values = {}
        for r in rr:
            t = r['job']['temperature']; m = maps[r['job']['id']]
            y = [m[s]['zero_z']['success_rate'] * 100 for s in grid]
            assert all(math.isfinite(v) for v in y)
            ax.plot([s / 1e6 for s in grid], y, color=colors[t], lw=2,
                    marker='o', ms=3.3, label=f'T={t:g}')
            values[str(t)] = m[global_step]['zero_z']['success_rate'] * 100
        waiting = [f'{t:g}' for t in temps if t not in [r['job']['temperature'] for r in rr]]
        ax.set_title(f'{task.title()} | seed {seed}', loc='left', fontweight='bold')
        ax.text(1, 1.02, f'Paired through {grid[-1]:,}', transform=ax.transAxes,
                ha='right', fontsize=8.5, color='#555555')
        ax.text(.02, .96, 'Queued: T=' + ', '.join(waiting), transform=ax.transAxes,
                va='top', fontsize=8, color='#555555',
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
        ax.axvline(global_step / 1e6, color='#555555', ls=':', lw=1, alpha=.5)
        ax.set_ylim(-3, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(alpha=.16)
        ax.set_xlim(0, xmax)
        if k == 0: ax.set_ylabel('Success rate (%)')
        if i == 3: ax.set_xlabel('Environment steps (M)')
        table.append(dict(task=task, seed=seed, steps=global_step, success_percent=values))
        panels.append(dict(task=task, seed=seed, common_steps=grid,
                           available_temperatures=[r['job']['temperature'] for r in rr]))
stamp = datetime.datetime.fromtimestamp(d['at'], datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
fig.suptitle('Adroit iBOLT temperature pilots | exact paired seeds and steps',
             x=.08, ha='left', fontsize=17, fontweight='bold', y=.985)
fig.text(.08, .955, '256 x 2 | critic LayerNorm | DACER on | fixed expert25 | z=0, conditional mean | 10 episodes',
         fontsize=10, color='#444444')
fig.legend(handles=[Line2D([0], [0], color=colors[t], lw=2.4, label=f'T={t:g}') for t in temps],
           loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False)
fig.text(.08, .016, f'{stamp} | No smoothing or interpolation. Each panel stops at its shared evaluation step.\n'
         f'Dotted line: {global_step:,} steps, common to all observed runs. Missing temperatures remain queued.',
         fontsize=9, color='#555555')
fig.tight_layout(rect=[.035, .055, .995, .91], h_pad=2)
fig.savefig(P / 'adroit-temperature-paired-seeds.png', dpi=180)
fig.savefig(P / 'adroit-temperature-paired-seeds.pdf')
summary = dict(at=d['at'], common_step=global_step, updates=maps[rows[0]['job']['id']][global_step]['updates'],
               evaluation='zero_z conditional mean; any success in 200 steps; 10 matched episodes',
               checks='Equal dataset hash per task, settings, update count, environment and policy evaluation seeds',
               table=table, panels=panels)
(P / 'paired-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
