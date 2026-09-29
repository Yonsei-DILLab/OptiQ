"""Post-hoc report from downloaded main-run results; never invokes training."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'tmp/reward-progress-worktree'
sys.path.insert(0, str(SOURCE))
from antmaze_experiments.progress_reward import maze_geometry
from antmaze_experiments.critic_diagnostics import route_label

COMMIT = '23603a7e7696aa64e8e49a38b986d6b430b6d6f3'
CONDITIONS = ('control', 'scale02', 'ema01', 'criticlr2', 'actordelay2', 'scale02ema01')
LABELS = ('Control', 'Reward & T / 5', 'EMA tau = .01', 'Critic LR x2', 'Actor every 2 updates', 'Scale / 5 + faster EMA')
COLORS = {'left': '#2879bc', 'right': '#e58629', 'upper': '#2879bc', 'lower': '#e58629',
          'uncommitted': '#929ca5', 'both': '#8b5cab'}
OUT = ROOT / 'report'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def initial_metric(diag, metric, stat='mean'):
    values = [(r['landmarks']['0'][metric]['normalized'][stat], r['episodes'])
              for r in diag['routes'].values()]
    if stat == 'rms':
        return float(np.sqrt(sum(v*v*n for v,n in values)/sum(n for _,n in values)))
    return float(sum(v*n for v,n in values)/sum(n for _,n in values))


def decomposition(diag, metric):
    return float(sum(r['initial_discounted_decomposition'][metric]['normalized']['mean'] * r['episodes']
                     for r in diag['routes'].values()) / diag['episodes'])


runs = {}
initial_hashes = {'actor': set(), 'critic': set()}
common_start = None
for host in sorted((ROOT/'results').glob('vast-heechan-*')):
    status = read(host/'status.json')
    assert not status['running'] and not status['pending'] and not status['failed']
    for directory in sorted((host/'runs').iterdir()):
        config = read(directory/'config.json')
        result = read(directory/'result.json')
        verify = read(directory/'dynamics-verification.json')
        assert result['completed'] and result['source_commit'] == COMMIT
        assert result['steps'] == 508416 and result['global_steps'] == 500224 and result['updates'] == 15632
        task, condition = config['task'], config['dynamics_profile']
        assert verify['actor_updates'] == (7816 if condition == 'actordelay2' else 15632)
        for model in initial_hashes:
            initial_hashes[model].add(verify['initial_parameters'][model]['sha256'])
        evaluations = []
        for destination in sorted((directory/'evaluations').glob('*/policy-fixed')):
            summary = read(destination/'summary.json')
            diag = read(destination/'critic-diagnostics.json')
            with np.load(destination/'rollouts.npz', allow_pickle=False) as a:
                xy, lengths = a['xy'].copy(), a['lengths'].copy()
                goals = a['goals'].copy()
                starts = a['initial_full_state'].copy()
                returns = a['returns'].copy()
            assert len(xy) == summary['episodes'] and diag['episodes'] == len(xy)
            assert summary['original_origin_fixed'] and summary['fixed']
            np.testing.assert_array_equal(starts, np.repeat(starts[:1], len(starts), axis=0))
            if common_start is None: common_start = starts[0]
            np.testing.assert_array_equal(starts[0], common_start)
            labels = [route_label(task, points[:int(n)+1])[0] for points,n in zip(xy,lengths)]
            counts = dict(Counter(labels))
            assert counts == {k:v['episodes'] for k,v in diag['routes'].items()}
            assert np.isclose((goals > 0).mean(), summary['success_rate'])
            assert np.isclose(returns.mean(), summary['mean_return'])
            evaluations.append(dict(step=summary['step'], global_step=summary['step']-8192,
                episodes=len(xy), counts=counts, success=summary['success_rate'],
                xy=xy, lengths=lengths, goals=goals, labels=labels, diag=diag,
                q=initial_metric(diag,'q_online'), mc=initial_metric(diag,'raw_mc'),
                mc_bootstrap=initial_metric(diag,'bootstrap_mc'),
                mc_minus_q=initial_metric(diag,'raw_mc_minus_q'),
                fit_rms=initial_metric(diag,'td_fit_residual','rms'),
                lag=initial_metric(diag,'target_tracking_term')))
        assert [e['episodes'] for e in evaluations] == [40]*5+[100]
        replay_path = sorted((directory/'replay-diagnostics').glob('*/summary.json'))[-1]
        replay = read(replay_path)
        assert replay['training_states_and_rng_unchanged']
        runs[task,condition] = dict(path=directory, config=config, result=result,
                                    evaluations=evaluations, replay=replay)
assert set(runs) == {(t,c) for t in ('v3','v4') for c in CONDITIONS}
assert all(len(h)==1 for h in initial_hashes.values())

records=[]
for (task,condition), run in runs.items():
    for e in run['evaluations']:
        records.append(dict(task=task, condition=condition, total_step=e['step'],
            global_step=e['global_step'], episodes=e['episodes'], success_rate=e['success'],
            **{k:e['counts'].get(k,0) for k in COLORS},
            q_normalized=e['q'], mc_normalized=e['mc'], bootstrap_mc_normalized=e['mc_bootstrap'],
            mc_minus_q_normalized=e['mc_minus_q'], initial_td_fit_rms=e['fit_rms'],
            initial_target_lag=e['lag'],
            **{'discounted_'+k:decomposition(e['diag'],k) for k in
               ('td_fit_residual','target_tracking_term','target_twin_min_term')}))
with (OUT/'evaluation_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
final=[r for r in records if r['total_step']==508416]
report=dict(source_commit=COMMIT, experiment='12 fresh policies; v3/v4 x 6 conditions; seed0',
    completed=12, running=0, pending=0, failed=0, post_warmup_steps=500224,
    full_policy='fresh latent plus conditional sigma; fixed original full state; no external DACER noise',
    final_evaluation_episodes=100, initial_parameters_identical=True, initial_full_state_identical=True,
    units='Q/reward/error all divided by reward_multiplier; common original reward units',
    final=final, history=records,
    limitations=['One training seed per condition; 100 evaluations are not 100 training seeds.',
                 'Gate entry counts are not successful goal-reaching route counts.',
                 'Different conditions follow different policies/state visitation distributions.',
                 'Raw MC ends at timeout; bootstrap MC is not independent ground truth.',
                 'Raw replay/checkpoints remain on the servers; this report archives evaluation NPZs and lightweight metadata.'])
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
for task in ('v3','v4'):
    walls,goals,bounds=maze_geometry(task)
    fig,axs=plt.subplots(2,3,figsize=(13,9),layout='constrained')
    for ax,condition,title in zip(axs.flat,CONDITIONS,LABELS):
        e=runs[task,condition]['evaluations'][-1]
        for x0,y0,x1,y1 in walls:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e4e7ea',edgecolor='#bbc2c9',lw=.5))
        for xy,n,label,goal in zip(e['xy'],e['lengths'],e['labels'],e['goals']):
            p=xy[:int(n)+1]; color=COLORS[label]
            ax.plot(p[:,0],p[:,1],color=color,alpha=.13,lw=.65)
            ax.scatter(*p[-1],s=6,color=color,alpha=.22)
        ax.scatter(*goals.T,marker='*',s=180,color='#31a05c',edgecolor='white',zorder=5)
        ax.scatter(0,0,s=26,c='black',zorder=6)
        short={'left':'L','right':'R','upper':'U','lower':'D','uncommitted':'none','both':'both'}
        counts=' / '.join(f'{short[k]} {v}' for k,v in e['counts'].items())
        ax.set_title(f'{title}\n{counts}; success {round(e["success"]*100)}/100',fontsize=11)
        ax.set_xlim(bounds[[0,2]]);ax.set_ylim(bounds[[1,3]]);ax.set_aspect('equal')
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle(f'{task.upper()} | 500k post-warmup | 100 direct-policy rollouts per condition',fontsize=15)
    fig.savefig(OUT/f'{task}_final_trajectories.png',dpi=170)
    plt.close(fig)

fig,axs=plt.subplots(2,6,figsize=(19,6.5),sharey=True,layout='constrained')
for row,task in enumerate(('v3','v4')):
    for col,(condition,title) in enumerate(zip(CONDITIONS,LABELS)):
        ax=axs[row,col]; es=runs[task,condition]['evaluations']; bottom=np.zeros(6)
        directions=('left','right','both','uncommitted') if task=='v3' else ('upper','lower','uncommitted')
        for direction in directions:
            values=np.array([e['counts'].get(direction,0)/e['episodes'] for e in es])*100
            ax.bar(np.arange(6),values,bottom=bottom,color=COLORS[direction],width=.8)
            bottom+=values
        ax.set_title(f'{task.upper()} | {title}',fontsize=10)
        ax.set_xticks(np.arange(6),['100','200','300','400','500','508'],rotation=55)
        ax.set_ylim(0,100);ax.set_xlabel('Total transitions (k)')
        if col==0:ax.set_ylabel('Episode share (%)')
fig.suptitle('Route use over training | v3: blue=left, orange=right; v4: blue=upper, orange=lower; gray=no gate entry\n40 episodes / checkpoint; final 508k=100 episodes. Fixed original start; direct full policy.',fontsize=13)
fig.savefig(OUT/'route_evolution.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
line_colors=plt.cm.tab10(np.arange(6))
for row,task in enumerate(('v3','v4')):
    for condition,title,color in zip(CONDITIONS,LABELS,line_colors):
        es=runs[task,condition]['evaluations']; x=[e['global_step']/1000 for e in es]
        for col,key in enumerate(('mc_minus_q','lag','fit_rms')):
            axs[row,col].plot(x,[e[key] for e in es],marker='o',ms=3,label=title,color=color)
    for col,title in enumerate(('MC return - online Q at start','One-step target tracking contribution at start','One-step TD fit residual RMS at start')):
        ax=axs[row,col];ax.set_title(task.upper()+': '+title,fontsize=11)
        ax.set_xlabel('Post-warmup transitions (k)');ax.set_ylabel('Original reward units');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8)
fig.suptitle('Critic diagnostics | all scales converted to original reward units\nCurrent-policy rollouts; across-condition differences also reflect different learned behavior.',fontsize=13)
fig.savefig(OUT/'critic_diagnostics.png',dpi=170);plt.close(fig)
print(json.dumps({'output':str(OUT),'final':final},indent=2))
