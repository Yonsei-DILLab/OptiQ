"""Validate and visualize one-seed OptiQ temperature sweep from saved data."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report'
OUT.mkdir(exist_ok=True)
MAZE_SOURCE = ROOT.parents[1] / 'antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py'
TEMPS = [1., .5, .25, .1, .05, .025, .01, .005]
HOSTS = {1.: 'vast-heechan-180', .25: 'vast-heechan-180',
         .05: 'vast-heechan-180', .01: 'vast-heechan-180',
         .5: 'vast-heechan-180-shadow', .1: 'vast-heechan-180-shadow',
         .025: 'vast-heechan-180-shadow', .005: 'vast-heechan-180-shadow'}
GOAL = np.array([-8., 0.])

class Markers(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.Constant({'R': 'r', 'G': 'g'}[node.id])

tree = ast.parse(MAZE_SOURCE.read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'MAZE_v1' for x in node.targets):
        maze = ast.literal_eval(Markers().visit(node.value))
        break
else:
    raise AssertionError('MAZE_v1 missing')
rr, cc = next((i, j) for i, row in enumerate(maze) for j, x in enumerate(row) if x == 'r')

def decorate(ax):
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(Rectangle(((j-cc)*4-2, (i-rr)*4-2), 4, 4,
                                       facecolor='#e3e8ee', edgecolor='#c3ccd5', lw=.6, zorder=0))
    ax.add_patch(Circle(GOAL, .5, color='#f5b537', alpha=.25, zorder=4))
    ax.scatter(*GOAL, marker='*', s=120, c='#d38b0b', zorder=5)
    ax.scatter(0, 0, marker='^', s=36, c='#172734', zorder=5)
    ax.set_xlim(-cc*4-2, (len(maze[0])-1-cc)*4+2)
    ax.set_ylim((len(maze)-1-rr)*4+2, -rr*4-2)
    ax.set_aspect('equal')
    ax.grid(alpha=.1)

def rollout(path):
    with np.load(path, allow_pickle=False) as file:
        d = {key: file[key].copy() for key in file.files}
    xy, length, goals = d['xy'], d['lengths'], d['goals']
    assert xy.shape[0] == len(length) == len(goals) == 100
    for line, n in zip(xy, length):
        assert np.isfinite(line[:int(n)+1]).all()
        assert np.isnan(line[int(n)+1:]).all()
    reached = np.nanmin(np.linalg.norm(xy-GOAL, axis=2), axis=1) <= .50002
    assert np.array_equal(reached, goals > 0)
    if bool(d['fixed']):
        assert np.all(d['initial_full_state'] == d['initial_full_state'][0])
    valid = xy[np.isfinite(xy).all(axis=2)]
    return d, dict(successes=int(reached.sum()),
        closest_goal=float(np.nanmin(np.linalg.norm(xy-GOAL,axis=2))),
        upper_entries=int(np.any((xy[:,:,0]<-2)&(xy[:,:,1]<-2),axis=1).sum()),
        lower_entries=int(np.any((xy[:,:,0]<-2)&(xy[:,:,1]>2),axis=1).sum()),
        past_obstacle=int(np.any(xy[:,:,0]<-6,axis=1).sum()),
        visited_bins=int(len(np.unique(np.floor(valid/.5).astype(np.int32),axis=0))),
        x_min=float(valid[:,0].min()), x_max=float(valid[:,0].max()),
        y_min=float(valid[:,1].min()), y_max=float(valid[:,1].max()))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
rows = []
data = {}
starts = []
for t in TEMPS:
    key = f'v1-optiq-T{t:g}-s0'
    run = ROOT / HOSTS[t] / 'runs' / key
    cfg = json.loads((run/'config.json').read_text())
    result = json.loads((run/'result.json').read_text())
    assert cfg['temperature'] == cfg['native']['alg']['actor']['temperature'] == t
    assert cfg['steps'] == result['steps'] == 328192
    assert cfg['expected_updates'] == result['updates'] == 10000
    xy = np.load(run/'training-xy.npy', mmap_mode='r')
    assert xy.shape == (328192, 2) and np.isfinite(xy).all()
    cells, first, counts = np.unique(np.floor(xy/.5).astype(np.int32), axis=0,
                                     return_index=True, return_counts=True)
    training = dict(visited_bins=int(len(cells)),
        closest_goal=float(np.linalg.norm(xy-GOAL, axis=1).min()),
        successes=result['training_successes'],
        upper_visits=int(np.sum((xy[:,0]<-2)&(xy[:,1]<-2))),
        lower_visits=int(np.sum((xy[:,0]<-2)&(xy[:,1]>2))),
        past_obstacle_visits=int(np.sum(xy[:,0]<-6)),
        top10_bin_fraction=float(np.sort(counts)[-10:].sum()/len(xy)))
    modes = {}
    for mode in ('policy-fixed','native-fixed','policy-natural'):
        path = run/'evaluations'/'0000328192'/mode/'rollouts.npz'
        d, metrics = rollout(path)
        modes[mode] = (d, metrics)
    starts.append(modes['policy-fixed'][0]['initial_full_state'][0])
    rows.append(dict(temperature=t,host=HOSTS[t],training=training,
                     policy_fixed=modes['policy-fixed'][1],
                     mu_only_fixed=modes['native-fixed'][1],
                     policy_natural=modes['policy-natural'][1]))
    data[t] = dict(run=run,cells=cells,first=first,counts=counts,modes=modes)
assert all(np.array_equal(starts[0], x) for x in starts[1:]), 'different fixed initial states across runs'
(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
with (OUT/'comparison.csv').open('w',newline='') as file:
    fields=['temperature','training_bins','training_successes','training_goal_distance',
            'training_upper_visits','training_lower_visits','policy_successes',
            'policy_goal_distance','policy_upper_entries','policy_lower_entries',
            'policy_past_obstacle','policy_bins','mu_only_upper_entries','mu_only_lower_entries']
    writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
    for r in rows:
        a,b,c=r['training'],r['policy_fixed'],r['mu_only_fixed']
        writer.writerow(dict(temperature=r['temperature'],training_bins=a['visited_bins'],
            training_successes=a['successes'],training_goal_distance=a['closest_goal'],
            training_upper_visits=a['upper_visits'],training_lower_visits=a['lower_visits'],
            policy_successes=b['successes'],policy_goal_distance=b['closest_goal'],
            policy_upper_entries=b['upper_entries'],policy_lower_entries=b['lower_entries'],
            policy_past_obstacle=b['past_obstacle'],policy_bins=b['visited_bins'],
            mu_only_upper_entries=c['upper_entries'],mu_only_lower_entries=c['lower_entries']))

for mode, title, filename in (
    ('policy-fixed','Direct stochastic policy (random z + conditional sigma)','policy_fixed_trajectories.png'),
    ('native-fixed','Random-z mu-only evaluation (conditional sigma omitted)','mu_only_fixed_trajectories.png')):
    fig, axes = plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,t in zip(axes.flat,TEMPS):
        d,m = data[t]['modes'][mode]
        for line,n in zip(d['xy'],d['lengths']):
            line=line[:int(n)+1]
            ax.plot(line[:,0],line[:,1],lw=.55,alpha=.2,color='#2476b4',zorder=2)
            ax.scatter(*line[-1],s=4,c='#dc604c',alpha=.45,zorder=3)
        decorate(ax)
        ax.set_title(f'T={t:g} · success {m["successes"]}/100 · goal {m["closest_goal"]:.1f}m\n'
                     f'upper/lower corridor {m["upper_entries"]}/{m["lower_entries"]}',fontsize=9)
    fig.suptitle('AntMaze v1 · 10k learner updates (328,192 env interactions) · seed 0\n'
                 +title+' · 100 rollouts from identical full state',fontsize=13)
    fig.supxlabel('Black triangle=start · gold star=goal · orange dots=episode endpoints',fontsize=9)
    fig.savefig(OUT/filename,dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
for ax,t in zip(axes.flat,TEMPS):
    rec=data[t]
    centers=(rec['cells']+.5)*.5
    mesh=ax.scatter(centers[:,0],centers[:,1],c=rec['counts']/328192,s=12,
                    marker='s',cmap='magma',norm=LogNorm(vmin=1/328192,vmax=.1),zorder=2)
    decorate(ax)
    a=next(r['training'] for r in rows if r['temperature']==t)
    ax.set_title(f'T={t:g} · {a["visited_bins"]} visited 0.5m bins\n'
                 f'goal {a["closest_goal"]:.1f}m · upper/lower visits {a["upper_visits"]}/{a["lower_visits"]}',fontsize=9)
fig.colorbar(mesh,ax=axes,label='Fraction of all training transitions per 0.5m bin',shrink=.75)
fig.suptitle('Behavior-policy training coverage · same 328,192 interactions per temperature',fontsize=13)
fig.savefig(OUT/'training_coverage.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
for t in TEMPS:
    first=np.sort(data[t]['first'])
    ax.step(np.r_[0,first+1,328192]/1000,np.r_[0,np.arange(1,len(first)+1),len(first)],
            where='post',lw=1.5,label=f'T={t:g}')
ax.set_xlabel('Environment interactions (thousands)');ax.set_ylabel('Distinct 0.5m bins visited')
ax.grid(alpha=.2);ax.legend(ncol=2)
ax.set_title('Cumulative training exploration · AntMaze v1 · seed 0')
fig.savefig(OUT/'coverage_curves.png',dpi=180);plt.close(fig)
print(json.dumps(dict(report=str(OUT),results=rows)))
