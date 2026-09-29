"""Analyze existing final v2 rollouts; no training or new evaluation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from antmaze_experiments.dense_off_report import GOALS, geometry

raw = ROOT/'raw'
expected = json.loads((ROOT/'remote-sha256.json').read_text())
for name, digest in expected.items():
    assert hashlib.sha256((raw/name).read_bytes()).hexdigest() == digest, name
cfg = json.loads((raw/'config.json').read_text())
assert cfg['source_commit'] == '8bb0c50a1356dc082106393f161ff3fc3d2de0af'
assert cfg['temperature'] == .01 and cfg['reward_profile'] == 'sparse'
assert cfg['noveld_coefficient'] == .01
training = json.loads((raw/'training-successes.json').read_text())
out = {'source': cfg['source_commit'], 'step': 3008256, 'seed': 0,
       'sha256_verified_files': len(expected),
       'training_goal_counts': {str(g): sum(t['goal']==g for t in training) for g in (1,2)},
       'route_rule': 'Successful route through central corridor to G2; left gate x<-4; right vertical branches x>6 and abs(y)>2.',
       'modes': {}}
data = {}
for p in sorted((raw/'evaluations/0003008256').glob('*/rollouts.npz')):
    with np.load(p, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    xy, lengths, goals = d['xy'], d['lengths'], d['goals']
    assert len(xy) == 100 and int(d['env_steps']) == 3008256
    for path, n in zip(xy, lengths):
        assert np.isfinite(path[:n+1]).all() and np.isnan(path[n+1:]).all()
    distances = np.linalg.norm(xy[:,:,None,:]-np.array(GOALS['v2'])[None,None,:,:],axis=-1)
    assert np.array_equal(np.nanmin(distances,axis=(1,2)) <= .50002, goals>0)
    assert np.array_equal(d['returns'], np.where(goals==1,20.,np.where(goals==2,10.,0.)))
    starts = len(np.unique(d['initial_full_state'],axis=0))
    assert starts == (1 if bool(d['fixed']) else 100)
    success = goals > 0
    left = np.any(xy[:,:,0] < -4, axis=1)
    branches = np.any((xy[:,:,0]>6)&(np.abs(xy[:,:,1])>2),axis=1)
    out['modes'][p.parent.name] = dict(episodes=100, successes=int(success.sum()),
        goal_counts={str(g):int((goals==g).sum()) for g in (0,1,2)},
        distinct_initial_states=starts,
        successful_left_entries=int((left&success).sum()),
        failed_left_entries=int((left&~success).sum()),
        successful_right_vertical_branch_entries=int((branches&success).sum()),
        successful_y_range=[float(np.nanmin(xy[success,:,1])),float(np.nanmax(xy[success,:,1]))])
    data[p.parent.name] = d
out['initial_states_equal_across_policy_and_native'] = {
    label: bool(np.array_equal(data['policy-'+label]['initial_full_state'], data['native-'+label]['initial_full_state']))
    for label in ('natural','fixed')}

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
fig, axes = plt.subplots(1,2,figsize=(12,6.7))
fig.subplots_adjust(left=.06,right=.98,top=.79,bottom=.18,wspace=.18)
maze, rr, cc = geometry('v2')
for ax,label in zip(axes, ('policy-natural','policy-fixed')):
    d=data[label]; m=out['modes'][label]
    for i,row in enumerate(maze):
        for j,cell in enumerate(row):
            if cell==1:
                ax.add_patch(Rectangle(((j-cc)*4-2,(i-rr)*4-2),4,4,
                    facecolor='#e2e7ed',edgecolor='#c9d1da',lw=.6))
    ax.add_patch(Rectangle((-2,-2),4,4,facecolor='#f1f5f7',lw=0))
    for ok in (False,True):
        for path,n,g in zip(d['xy'],d['lengths'],d['goals']):
            if bool(g>0)!=ok: continue
            line=path[:n+1]
            ax.plot(*line.T,c='#177da7' if ok else '#ba867a',lw=.85 if ok else .7,alpha=.32 if ok else .45)
            if not ok: ax.scatter(*line[-1],marker='x',s=18,c='#b96956',lw=.9,zorder=5)
    for i,g in enumerate(GOALS['v2'],1):
        ax.add_patch(Circle(g,.5,facecolor='#e9b548',alpha=.4,zorder=5))
        ax.scatter(*g,marker='*',s=170,c='#e6b12f',edgecolor='#946716',lw=.7,zorder=8)
        ax.annotate(f'G{i}: {m["goal_counts"][str(i)]}/100',g,xytext=(0,18),textcoords='offset points',
                    ha='center',fontsize=11,fontweight='bold',color='#72500a',zorder=9)
    ax.scatter(d['xy'][:,0,0],d['xy'][:,0,1],marker='^',s=20,c='#20384a',
               edgecolor='white',lw=.3,alpha=.7,zorder=7)
    ax.set_xlim(-14,14);ax.set_ylim(14,-14);ax.set_aspect('equal')
    ax.set_xticks([-8,0,8]);ax.set_yticks([-8,0,8]);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.set_title(('100 random starts' if label.endswith('natural') else '100 identical full-state starts')+
                 f'\nSuccess {m["successes"]}/100; one successful corridor',fontsize=12,pad=10)
fig.suptitle('AntMaze v2 | OptiQ | 3,008,256 interactions | seed 0\n'
             'All successful rollouts go right to G2',fontsize=16,y=.97)
handles=[Line2D([],[],c='#177da7',lw=2,label='Success to G2'),
         Line2D([],[],c='#ba867a',lw=2,label='Failed rollout'),
         Line2D([],[],marker='^',c='#20384a',ls='',label='Start')]
fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.08),frameon=False)
fig.text(.5,.035,'Direct policy: random z + conditional sigma. No extra DACER noise.\n'
         'Sparse reward + NovelD 0.01; T=0.01. All 100 episodes per panel shown, including failures.',
         ha='center',fontsize=9,color='#4b5863')
fig.savefig(ROOT/'v2_final_routes.png',dpi=180,facecolor='white')
fig.savefig(ROOT/'v2_final_routes.pdf',facecolor='white')
plt.close(fig)
(ROOT/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
