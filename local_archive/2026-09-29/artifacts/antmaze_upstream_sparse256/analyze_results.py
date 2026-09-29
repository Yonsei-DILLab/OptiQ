"""Analyze saved sparse256 outputs only; never modify or restart training."""
from pathlib import Path
import ast
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / 'report'
OUT.mkdir(exist_ok=True)
GOALS = {'v1': [(-8, 0)], 'v2': [(-8, 8), (8, 0)],
         'v3': [(-12, 12), (12, -12)], 'v4': [(-16, 4), (-16, -4)]}
COLORS = {'optiq': '#1683bf', 'mfpo': '#8d4ab5', 'dipo': '#168b77'}
LABELS = {'optiq': 'OptiQ', 'mfpo': 'MFPO', 'dipo': 'DIPO'}
SOURCE = '0ebd8d26c711d79723f457343b36eb8788bd87b8'

class ReplaceMarkers(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.Constant({'R': 'r', 'G': 'g'}[node.id])

tree = ast.parse((REPO / 'antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py').read_text())
MAPS = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ['MAZE_' + x for x in GOALS]:
            MAPS[name[5:]] = ast.literal_eval(ReplaceMarkers().visit(node.value))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})

def load(path):
    return json.loads(path.read_text())

def geometry(task):
    maze = MAPS[task]
    rr, cc = next((i, j) for i, row in enumerate(maze) for j, x in enumerate(row) if x == 'r')
    return maze, rr, cc

def decorate(ax, task, start=None):
    maze, rr, cc = geometry(task)
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle(((j-cc)*4-2, (i-rr)*4-2), 4, 4,
                    facecolor='#e0e5eb', edgecolor='#cbd1d9', linewidth=.7, zorder=0))
    for i, g in enumerate(GOALS[task]):
        ax.add_patch(Circle(g, .5, color='#eead22', alpha=.22, zorder=4))
        ax.scatter(*g, marker='*', s=120, color='#eead22', edgecolor='#845e10', linewidth=.6, zorder=5)
        ax.text(g[0]+.55, g[1]-.2, f'G{i+1}', color='#845e10', fontsize=9, zorder=6)
    if start is not None:
        ax.scatter(*start, marker='^', s=60, color='#162630', edgecolor='white', linewidth=.5, zorder=7)
    ax.set_xlim(-cc*4-2, (len(maze[0])-1-cc)*4+2)
    ax.set_ylim((len(maze)-1-rr)*4+2, -rr*4-2)
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m; map row direction)')

def rollout_metrics(path, task):
    with np.load(path, allow_pickle=False) as data:
        d = {k: data[k].copy() for k in data.files}
    xy, goals, lens = d['xy'], d['goals'], d['lengths']
    assert xy.shape[0] == len(goals) == len(lens)
    assert np.isfinite(d['initial_full_state']).all()
    for i, length in enumerate(lens):
        assert np.isfinite(xy[i, :int(length)+1]).all()
        assert np.isnan(xy[i, int(length)+1:]).all()
    dist = np.stack([np.linalg.norm(xy-np.array(g), axis=-1) for g in GOALS[task]], axis=-1)
    observed = np.nanmin(dist, axis=(1, 2)) <= .50002
    assert np.array_equal(observed, goals > 0), path
    expected_returns = np.where(goals == 0, 0, np.where((task == 'v2') & (goals == 1), 20, 10))
    assert np.array_equal(expected_returns, d['returns']), path
    fixed = bool(d['fixed'])
    if fixed: assert np.all(d['initial_full_state'] == d['initial_full_state'][0])
    displacement = np.linalg.norm(xy-xy[:, :1], axis=-1)
    valid = xy[np.isfinite(xy).all(axis=-1)]
    m = dict(episodes=len(goals), successes=int((goals > 0).sum()),
        success_rate=float(np.mean(goals > 0)), mean_return=float(d['returns'].mean()),
        fixed_full_state=fixed, step=int(d['env_steps']), mode=str(d['mode']),
        goal_counts={str(g): int((goals == g).sum()) for g in np.unique(goals)},
        min_goal_distance=float(np.nanmin(dist)),
        per_goal_min_distance=[float(np.nanmin(dist[:, :, j])) for j in range(len(GOALS[task]))],
        median_max_displacement=float(np.median(np.nanmax(displacement, axis=1))),
        median_path_length=float(np.median(np.nansum(np.linalg.norm(np.diff(xy, axis=1), axis=2), axis=1))),
        visited_bins_05m=len(np.unique(np.floor(valid/.5).astype(int), axis=0)),
        xy_min=valid.min(0).tolist(), xy_max=valid.max(0).tolist())
    if task == 'v1':
        m['upper_corridor_entry_count'] = int(np.any((xy[:,:,0] < -2) & (xy[:,:,1] < -2), axis=1).sum())
        m['lower_corridor_entry_count'] = int(np.any((xy[:,:,0] < -2) & (xy[:,:,1] > 2), axis=1).sum())
        m['goal_side_x_below_minus6_count'] = int(np.any(xy[:,:,0] < -6, axis=1).sum())
    if task == 'v2':
        m['left_x_below_minus4_count'] = int(np.any(xy[:,:,0] < -4, axis=1).sum())
        m['right_x_above4_count'] = int(np.any(xy[:,:,0] > 4, axis=1).sum())
    return d, m

RUNS = {}
for run in sorted((ROOT/'results').glob('*/runs/*')):
    if not (run/'config.json').exists(): continue
    cfg = load(run/'config.json'); task, method = cfg['task'], cfg['method']
    assert cfg['source_commit'] == SOURCE
    result = load(run/'result.json') if (run/'result.json').exists() else None
    failure = load(run/'failure.json') if (run/'failure.json').exists() else None
    record = dict(task=task, method=method, seed=cfg['seed'], source=SOURCE,
        status='completed' if result else 'failed', result=result, failure=failure,
        last_progress=load(run/'progress.json'), evaluations={})
    for p in sorted((run/'evaluations').glob('*/*/rollouts.npz')):
        d, metrics = rollout_metrics(p, task)
        summary = load(p.with_name('summary.json'))
        assert metrics['success_rate'] == summary['success_rate']
        record['evaluations'][str(p.parent.relative_to(run/'evaluations'))] = metrics
    if result:
        assert result['steps'] == cfg['steps'] and result['updates'] == cfg['expected_updates']
        assert result['checkpoint']['readback_verified'] and result['checkpoint']['sparse_replay_verified']
        xy = np.load(run/'training-xy.npy', mmap_mode='r')
        assert xy.shape == (cfg['steps'], 2) and np.isfinite(xy).all()
        cells, first, counts = np.unique(np.floor(xy/.5).astype(np.int32), axis=0,
            return_index=True, return_counts=True)
        distances = [float(np.linalg.norm(xy-np.array(g), axis=1).min()) for g in GOALS[task]]
        successes = load(run/'training-successes.json')
        assert len(successes) == result['training_successes']
        if not successes: assert min(distances) > .5
        record['training'] = dict(steps=len(xy), successes=len(successes),
            episodes=result['training_episodes'], visited_bins_05m=len(cells),
            min_goal_distance=min(distances), per_goal_min_distance=distances,
            xy_min=xy.min(0).tolist(), xy_max=xy.max(0).tolist(),
            radius95=float(np.quantile(np.linalg.norm(xy, axis=1), .95)),
            top10_bin_fraction=float(np.sort(counts)[-10:].sum()/len(xy)))
        np.savez_compressed(OUT/(run.name+'-coverage.npz'), cells=cells, first=first, counts=counts)
    RUNS[run.name] = record

def path_for(key, label, step=None):
    host = '180' if RUNS[key]['task'] in ('v1', 'v3') else '199'
    base = ROOT/'results'/host/'runs'/key/'evaluations'
    if step is None: step = RUNS[key]['result']['steps']
    return base/f'{step:010d}'/label/'rollouts.npz'

def plot_trajectories(mode):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.8), constrained_layout=True)
    for i, task in enumerate(('v1','v2')):
        for j, method in enumerate(('optiq','mfpo')):
            key = f'{task}-{method}-s0'; d, m = rollout_metrics(path_for(key, mode+'-fixed'), task)
            ax = axes[i,j]
            for line, length in zip(d['xy'],d['lengths']):
                line=line[:int(length)+1]
                ax.plot(line[:,0],line[:,1],color=COLORS[method],alpha=.2,lw=.65)
                ax.scatter(*line[-1],s=7,color='#db6150',alpha=.45,zorder=3)
            decorate(ax, task, d['xy'][0,0])
            ax.set_title(f'{task.upper()} · {LABELS[method]} | success {m["successes"]}/100\n'
                f'closest goal {m["min_goal_distance"]:.2f} m · median max displacement {m["median_max_displacement"]:.2f} m')
    subtitle = 'Direct policy draws (OptiQ includes conditional sigma)' if mode=='policy' else 'Native evaluation (OptiQ mu-only; MFPO Q-best-of-10)'
    fig.suptitle('Final AntMaze trajectories · 3.008M interactions · seed 0\n'+subtitle+
        '\n100 rollouts from one identical full state per policy; all failures shown',fontsize=13)
    fig.supxlabel('Triangle: initial state   Star/circle: goal / 0.5 m success radius   Coral dot: failed endpoint',fontsize=9)
    fig.savefig(OUT/f'final_{mode}_fixed_trajectories.png',dpi=175)
    plt.close(fig)

plot_trajectories('policy'); plot_trajectories('native')

fig, axes = plt.subplots(2,2,figsize=(11,9.8),constrained_layout=True)
for i,task in enumerate(('v1','v2')):
    for j,method in enumerate(('optiq','mfpo')):
        key=f'{task}-{method}-s0'; cov=np.load(OUT/(key+'-coverage.npz'))
        ax=axes[i,j];maze,rr,cc=geometry(task)
        xe=np.arange(-cc*4-2,(len(maze[0])-1-cc)*4+2+.25,.5)
        ye=np.arange(-rr*4-2,(len(maze)-1-rr)*4+2+.25,.5)
        grid=np.zeros((len(xe)-1,len(ye)-1))
        centers=(cov['cells']+.5)*.5
        ix=np.searchsorted(xe,centers[:,0],side='right')-1
        iy=np.searchsorted(ye,centers[:,1],side='right')-1
        assert ((ix>=0)&(ix<grid.shape[0])&(iy>=0)&(iy<grid.shape[1])).all()
        grid[ix,iy]=cov['counts']/RUNS[key]['training']['steps']
        mesh=ax.pcolormesh(xe,ye,np.ma.masked_equal(grid.T,0),cmap='magma',
            norm=LogNorm(vmin=1e-6,vmax=.1),zorder=2,rasterized=True)
        decorate(ax,task,(0,0))
        t=RUNS[key]['training']
        ax.set_title(f'{task.upper()} · {LABELS[method]} | {t["visited_bins_05m"]} visited 0.5 m bins\n'
            f'training successes {t["successes"]} · closest goal {t["min_goal_distance"]:.2f} m')
fig.colorbar(mesh,ax=axes,label='Fraction of all training transitions per 0.5 m bin',shrink=.78)
fig.suptitle('Training exploration · all 3.008M transitions · seed 0\nSparse reward + NovelD 0.01 | shared occupancy color scale',fontsize=13)
fig.supxlabel('Training occupancy includes behavior exploration; it is not the final-policy rollout distribution.',fontsize=9)
fig.savefig(OUT/'training_occupancy.png',dpi=175);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
for ax,task in zip(axes,('v1','v2')):
    for method in ('optiq','mfpo'):
        key=f'{task}-{method}-s0';cov=np.load(OUT/(key+'-coverage.npz'))
        first=np.sort(cov['first'])
        ax.step(np.r_[0,first+1,3008256]/1e6,np.r_[0,np.arange(1,len(first)+1),len(first)],
            where='post',label=LABELS[method],color=COLORS[method],lw=2)
    ax.set_title(task.upper());ax.set_xlabel('Environment interactions (million)')
    ax.set_ylabel('Unique visited 0.5 m bins');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Cumulative training exploration · 256 environments · seed 0',fontsize=13)
fig.savefig(OUT/'training_coverage_curves.png',dpi=175);plt.close(fig)

fig,axes=plt.subplots(1,4,figsize=(16,4.8),constrained_layout=True)
for ax,task in zip(axes,GOALS):
    key=f'{task}-dipo-s0';r=RUNS[key]
    available=[(int(k.split('/')[0]),v) for k,v in r['evaluations'].items() if k.endswith('policy-natural')]
    step,metrics=max(available)
    d,m=rollout_metrics(path_for(key,'policy-natural',step),task)
    for line,length in zip(d['xy'],d['lengths']):
        line=line[:int(length)+1];ax.plot(line[:,0],line[:,1],color=COLORS['dipo'],alpha=.55,lw=.8)
    decorate(ax,task,d['xy'][0,0])
    ax.set_title(f'{task.upper()} · last saved eval {step/1e6:.2f}M\n'
        f'{m["successes"]}/{m["episodes"]} success; failed at {r["failure"]["step"]/1e6:.3f}M')
fig.suptitle('DIPO partial results only · direct policy · 20 natural-start rollouts each\nDifferent budgets; CUDA BCE target assertion interrupted every run',fontsize=13)
fig.savefig(OUT/'dipo_partial_trajectories.png',dpi=175);plt.close(fig)

summary=dict(source=SOURCE,seed=0,completed=4,failed=4,unstarted_held=8,runs=RUNS,
    methodology=dict(grid_m=.5,primary_mode='policy-fixed',primary_episodes=100,
        natural_fixed_not_pooled=True,preflight_excluded=True,
        successes_from_raw_goal_ids_verified_against_xy=True,
        goal_distance='Euclidean, not obstacle-aware route length',
        interpretation='Corridor-entry counts describe partial exploration, not successful route modes'))
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
rows=[]
for key,r in RUNS.items():
    if r['status']!='completed':continue
    m=r['evaluations'][f'{r["result"]["steps"]:010d}/policy-fixed'];t=r['training']
    rows.append(dict(run=key,steps=t['steps'],training_successes=t['successes'],
        final_successes=m['successes'],final_episodes=m['episodes'],visited_bins_05m=t['visited_bins_05m'],
        closest_goal_training=t['min_goal_distance'],closest_goal_final=m['min_goal_distance'],
        median_max_displacement=m['median_max_displacement']))
with (OUT/'completed_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print(json.dumps(rows,indent=2))
print('Validated raw rollout files:',sum(len(r['evaluations']) for r in RUNS.values()))
