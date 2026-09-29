"""Verify and render the separately sampled, unchanged-checkpoint diagnostic."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from antmaze.multimodal.analysis import summarize
from antmaze.multimodal.dense_noveld_report import equivalent

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
TRAINING = '19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5'
DIAGNOSTIC = '642327b6c706ec78c35098eeb866d5758bd3377e'
METHODS = ['optiq', 'sac', 'mfpo', 'meow']
NAMES = dict(optiq='OptiQ', sac='SAC', mfpo='MFPO', meow='MEOW')
arrays, summaries, geometries, proofs, initial = {}, {}, {}, {}, {}
total = 0

for method in METHODS:
    folder = ROOT / 'rerollouts' / method
    result = read(folder / 'result.json')
    assert result['completed'] and result['training_unchanged']
    for task in (['v3', 'v1'] if method == 'optiq' else ['v3']):
        here = folder / task
        verification, provenance = read(here/'verification.json'), read(here/'provenance.json')
        assert verification['passed'] and verification['model_unchanged']
        assert provenance['training_source'] == verification['source_commit'] == TRAINING
        assert provenance['diagnostic_source'] == DIAGNOSTIC
        assert verification['checkpoint_step'] == 1000000 and verification['training_seed'] == 0
        assert provenance['script_sha256'] == sha(ROOT.parents[1]/'antmaze/route_diagnostics/rollout_audit.py')
        run = ROOT.parent/'antmaze_dense_noveld_1m/runs'/f'{task}-{method}-s0'
        archive = read(run/'archive-verification.json')
        assert archive['passed'] and archive['steps'] == 1000000
        assert provenance['checkpoint_sha256'] == archive['sha256']['resume/step_0001000000/state.pt']
        assert provenance['config'] == read(run/'config.json')
        geometry = provenance['config']['environment']
        if task in geometries:
            for k in ('walls','goals','horizon'): assert geometry[k] == geometries[task][k]
        geometries[task] = geometry
        for filename, digest in verification['npz_sha256'].items():
            assert sha(here/filename) == digest
        modes = ['policy','native','episode_latent_mu'] if method == 'optiq' and task == 'v3' else ['policy']
        for mode in modes:
            file = here/f'{mode}.npz'
            with np.load(file, allow_pickle=False) as z:
                a = {k:z[k] for k in z.files}
            s = read(here/f'{mode}.json')
            recomputed = summarize(task,a['xy'],a['lengths'],a['goal_ids'],a['returns'])
            equivalent(recomputed,s)
            equivalent(s,result['results'][task+'-'+mode])
            assert s['episodes'] == 100 and s['fixed_full_state']
            assert s['altered_policy_diagnostic'] == (mode == 'episode_latent_mu')
            assert not s['added_action_noise'] and not s['intrinsic_reward_in_evaluation']
            states = a['initial_simulator_state']
            np.testing.assert_array_equal(states,np.broadcast_to(states[0],states.shape))
            if task in initial: np.testing.assert_array_equal(initial[task],states[0])
            initial[task] = states[0]
            assert len(np.unique(a['policy_batch_seeds'])) == 10
            for i,n in enumerate(a['lengths']):
                assert 1 <= n <= geometry['horizon']
                assert np.isfinite(a['xy'][i,:n+1]).all()
                assert np.isfinite(a['observations'][i,:n+1]).all()
                assert np.isfinite(a['actions'][i,:n]).all()
                assert np.max(np.abs(a['actions'][i,:n])) <= 1.00001
                distance = np.linalg.norm(a['xy'][i,1:n+1,None,:]-np.array(geometry['goals']),axis=-1)
                np.testing.assert_allclose(a['returns'][i],-distance.min(axis=1).sum(),rtol=2e-6,atol=.005)
                if a['goal_ids'][i]: assert distance[-1,a['goal_ids'][i]-1] <= .50001
            key = f'{task}-{method}-{mode}'
            arrays[key], summaries[key] = a,s
            proofs[key] = dict(episodes=100, success=int(np.count_nonzero(a['goal_ids'])),
                goals=s['successful_goals']['counts'], routes=s['successful_routes']['counts'],
                failures=int(np.sum(a['goal_ids']==0)),
                unique_first_actions=int(len(np.unique(a['actions'][:,0],axis=0))),
                unique_step10_xy=int(len(np.unique(a['xy'][:,10],axis=0))),
                raw_sha256=sha(file), model_unchanged=True, fixed_full_state_verified=True)
            total += 100
assert total == 700

colors = dict(G1='#1686a6',G2='#d6802d',failure='#a6a6a6',upper='#1686a6',lower='#d6802d')
def panel(ax,key,title):
    task=key.split('-')[0]; g=geometries[task]; a=arrays[key];s=summaries[key]
    for x,y in g['walls']:ax.add_patch(Rectangle((x-2,y-2),4,4,color='#42484e',zorder=1))
    for path,n,route in zip(a['xy'],a['lengths'],s['routes']):
        color = 'failure' if route=='failure' else route.split('/')[1] if task=='v1' else route.split('/')[0]
        p=path[:int(n)+1]
        ax.plot(p[:,0],p[:,1],color=colors[color],lw=.85,alpha=.25 if color!='failure' else .55,zorder=3)
    ax.scatter(*a['xy'][0,0],marker='*',color='black',s=70,zorder=6)
    for i,(x,y) in enumerate(g['goals'],1):
        ax.add_patch(Circle((x,y),.5,color='#44a25f',zorder=5));ax.text(x,y+.9,f'G{i}',ha='center',fontsize=9,zorder=6)
    w=np.array(g['walls']);ax.set(xlim=(w[:,0].min()-2,w[:,0].max()+2),ylim=(w[:,1].min()-2,w[:,1].max()+2),aspect='equal',xlabel='x',ylabel='y')
    p=proofs[key]
    detail=f"G1 {p['goals'].get('1',0)}, G2 {p['goals'].get('2',0)}, fail {p['failures']}" if task!='v1' else f"Upper {p['routes'].get('G1/upper',0)}, lower {p['routes'].get('G1/lower',0)}, fail {p['failures']}"
    ax.set_title(title+'\n'+detail,fontsize=11)

def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(ROOT/f'{name}.{ext}',dpi=180)
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10.5,12.7))
for ax,m in zip(axes.flat,METHODS):panel(ax,f'v3-{m}-policy',NAMES[m]+' · direct stochastic policy')
for ax in axes[0]:ax.set_xlabel('')
fig.suptitle('AntMaze v3 · 100 new rollouts per trained policy\n1M interactions · training seed 0 · identical full initial state',fontsize=14)
fig.legend([Line2D([0],[0],color=colors[k],lw=2) for k in ('G1','G2','failure')],['Goal 1','Goal 2','Failure'],loc='lower center',bbox_to_anchor=(.5,.038),ncol=3,frameon=False)
fig.text(.5,.018,'OptiQ: random z + conditional sigma. No extra DACER noise or intrinsic reward in evaluation.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.075,1,.93),h_pad=3.5);save(fig,'v3-new-policy-rollouts')

fig,axes=plt.subplots(1,3,figsize=(14.5,6.3))
for ax,mode,label in zip(axes,['policy','native','episode_latent_mu'],['Fresh z + conditional sigma','Fresh z · mu-only','Held z per episode · mu-only\nAltered sampling diagnostic']):
    panel(ax,'v3-optiq-'+mode,label)
fig.suptitle('AntMaze v3 · OptiQ · same trained checkpoint, three sampling modes',fontsize=15)
fig.text(.5,.06,'100 new rollouts per panel, identical full initial state. Holding z does not recover the other goal.',ha='center',fontsize=10)
fig.text(.5,.025,'Held-z sampling changes the temporal policy; it is not a new training run or a replacement production score.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.92));save(fig,'optiq-v3-sampling-diagnostic')

fig,ax=plt.subplots(figsize=(8,5.8));panel(ax,'v1-optiq-policy','OptiQ v1 · 100 new direct-policy rollouts')
fig.suptitle('Positive control: two routes remain visible',fontsize=14)
fig.text(.5,.045,'Same full initial state · training seed 0 · 1M checkpoint · fresh z + conditional sigma',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.94));save(fig,'optiq-v1-two-route-control')

output=dict(passed=True,episodes=total,training_source=TRAINING,diagnostic_source=DIAGNOSTIC,
    checkpoint_sha_matches_verified_archives=True,identical_full_initial_state_within_each_task=True,
    summaries_recomputed=True,reward_recomputed=True,results=proofs)
(ROOT/'rerollout-verification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
