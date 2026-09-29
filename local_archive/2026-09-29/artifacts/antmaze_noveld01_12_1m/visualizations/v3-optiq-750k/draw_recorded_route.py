"""Visualize recorded route categories, never reconstruct missing trajectories."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parent
c = json.loads((ROOT / 'config.json').read_text())
records = {}
for mode in ('policy', 'native'):
    history = json.loads((ROOT / f'history-{mode}-natural.json').read_text())
    records[mode] = next(row for row in history if row['step'] == 750000)
assert c['source_commit'] == '0d23377f31bdd5c360e9c79d44bb9f4b1341012b'
assert records['policy']['successful_routes']['counts'] == {'G1/passage-y+4': 10}
assert records['native']['successful_routes']['counts'] == {'G1/passage-y+4': 9}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(12, 7.3), facecolor='#fafbfc')
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=.06, right=.97,
                      bottom=.16, top=.80, wspace=.20)
ax = fig.add_subplot(gs[0])
ax.set_facecolor('white')
teal, orange, wall = '#07877c', '#cc8032', '#454e5c'
for x, y in c['environment']['walls']:
    ax.add_patch(Rectangle((x-2, y-2), 4, 4, facecolor=wall, edgecolor='#fafbfc', linewidth=.4))
for i, (x, y) in enumerate(c['environment']['goals'], 1):
    color = teal if i == 1 else orange
    ax.add_patch(Circle((x, y), .5, color=color, zorder=4))
    ax.add_patch(Circle((x, y), 1.1, fill=False, edgecolor=color, linewidth=1.7, zorder=4))
    ax.text(x+1.6 if i == 1 else x-1.6, y, f'G{i}', color=color,
            ha='left' if i == 1 else 'right', va='center', weight='bold', zorder=5)
ax.scatter(0, 0, marker='*', s=160, c='#24344b', zorder=6)
ax.text(1.2, -.8, 'Start (0, 0)', fontsize=10, color='#24344b', zorder=6)
# The classifier stores the last leftward x=-8 crossing rounded to a 4m bin.
# Highlight the corresponding opening only; do not invent a path polyline.
ax.add_patch(Rectangle((-10, 2), 4, 4, color=teal, alpha=.22, zorder=3))
ax.plot([-8, -8], [2, 6], c=teal, linewidth=5, solid_capstyle='round', zorder=5)
ax.annotate('Recorded passage\nx = -8, y ≈ +4', xy=(-8, 4), xytext=(0, 10.5),
            fontsize=10, color=teal, ha='center',
            arrowprops={'arrowstyle': '-', 'color': teal, 'lw': 1.3}, zorder=7,
            bbox={'boxstyle': 'round,pad=.35', 'fc': 'white', 'ec': teal, 'alpha': .98})
ax.set(xlim=(-18, 18), ylim=(-18, 18), aspect='equal', xlabel='x (m)', ylabel='y (m)')
ax.set_xticks([-12, -8, 0, 8, 12]); ax.set_yticks([-12, -4, 0, 4, 12])
ax.tick_params(labelsize=9)
for spine in ax.spines.values(): spine.set_visible(False)

info = fig.add_subplot(gs[1]); info.axis('off')
info.text(0, .97, '750k evaluation records', fontsize=17, weight='bold', va='top', color='#182637')
for y, mode, title, subtitle in [(.80, 'policy', 'Direct policy sampling', 'Random z + conditional sigma'),
                                 (.45, 'native', 'Mu-only evaluation', 'Random z, no conditional sigma')]:
    r = records[mode]
    successes = sum(r['successful_routes']['counts'].values())
    info.text(0, y, title, weight='bold', fontsize=12, color='#182637')
    info.text(0, y-.06, subtitle, fontsize=10, color='#5a6677')
    for i in range(10):
        success = r['routes'][i] != 'failure'
        info.add_patch(Rectangle((i*.091, y-.17), .071, .062,
                                 transform=info.transAxes, facecolor=teal if success else '#d6dae0'))
    info.text(0, y-.235, f'{successes}/10 success · all successful runs use G1 / y+4',
              fontsize=10, color='#182637')
info.text(0, .055, 'G2 reached: 0/10 in both modes\nOne recorded successful route category',
          fontsize=11, color='#182637', linespacing=1.65)
fig.text(.06, .947, 'OptiQ · AntMaze v3 · 750k steps', fontsize=22, weight='bold', color='#182637')
fig.text(.06, .898, 'NovelD 0.1 · training seed 0 · passage and destination from saved evaluation summaries',
         fontsize=11, color='#5a6677')
fig.text(.06, .081, 'ROUTE-LOCATION DIAGRAM — actual trajectory coordinates were not saved.',
         fontsize=11, weight='bold', color='#9b472e')
fig.text(.06, .041, 'The highlighted gate and G1 are recorded; the movement between them and the failed path are unavailable.',
         fontsize=10, color='#5a6677')
fig.savefig(ROOT / 'recorded-route-location.png', dpi=180, facecolor=fig.get_facecolor())
fig.savefig(ROOT / 'recorded-route-location.pdf', facecolor=fig.get_facecolor())
plt.close(fig)
proof = {'step': 750000, 'source_commit': c['source_commit'],
         'type': 'recorded_route_category_location_not_raw_trajectories',
         'raw_trajectory_available': False, 'checkpoint_750k_available': False,
         'records': records,
         'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in ROOT.glob('*.json') if p.name != 'provenance.json'}}
(ROOT / 'provenance.json').write_text(json.dumps(proof, indent=2) + '\n')
print(ROOT / 'recorded-route-location.png')
