"""Latest saved v2 policy trajectories, with direction entry separate from success."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
from antmaze_experiments.dense_off_report import geometry, GOALS

fig,axes=plt.subplots(1,2,figsize=(12,6.8))
fig.subplots_adjust(left=.065,right=.97,top=.8,bottom=.19,wspace=.2)
colors=['#287fbb','#e58a2f','#8763a9','#89949e']
result={'source_commit':'cac1365fd9878d46874bbc09a840816ba1461499',
        'step':1750016,'seed':0,'entry_rule':'left x<-4; right x>4, any time in rollout', 'modes':{}}
maze,rr,cc=geometry('v2')
for ax,mode in zip(axes,['policy-natural','native-natural']):
    p=ROOT/'raw'/mode/'rollouts.npz'
    with np.load(p,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    xy,n,g=d['xy'],d['lengths'],d['goals']
    assert len(g)==40 and int(d['env_steps'])==1750016 and not bool(d['fixed'])
    assert len(np.unique(d['initial_full_state'],axis=0))==40
    for line,length in zip(xy,n):
        assert np.isfinite(line[:length+1]).all() and np.isnan(line[length+1:]).all()
    dist=np.linalg.norm(xy[:,:,None,:]-np.array(GOALS['v2'])[None,None],axis=-1)
    assert np.array_equal(np.nanmin(dist,axis=(1,2))<=.50002,g>0)
    assert np.array_equal(d['returns'],np.where(g==1,20.,np.where(g==2,10.,0.)))
    left=np.any(xy[:,:,0]<-4,axis=1);right=np.any(xy[:,:,0]>4,axis=1)
    cls=np.where(left&right,2,np.where(left,0,np.where(right,1,3)))
    metrics=dict(episodes=40,left=int(left.sum()),right=int(right.sum()),both=int((left&right).sum()),
        neither=int((~left&~right).sum()),goals={str(k):int((g==k).sum()) for k in [0,1,2]},
        toward_left_lower_goal=int(np.any((xy[:,:,0]<-6)&(xy[:,:,1]>2),axis=1).sum()),
        nearest_goal_m=np.nanmin(dist,axis=(0,1)).tolist(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    result['modes'][mode]=metrics
    for i,row in enumerate(maze):
        for j,cell in enumerate(row):
            if cell==1:ax.add_patch(Rectangle(((j-cc)*4-2,(i-rr)*4-2),4,4,facecolor='#e3e8ed',edgecolor='#cbd3dc',lw=.6))
    for path,length,k,goal in zip(xy,n,cls,g):
        line=path[:length+1]
        ax.plot(*line.T,color=colors[k],alpha=.6,lw=.8)
        ax.scatter(*line[-1],marker='o' if goal else 'x',s=18,c='#22925f' if goal else '#b35b53',lw=.8,zorder=5)
    ax.scatter(xy[:,0,0],xy[:,0,1],marker='^',s=20,color='#253b49',edgecolor='white',lw=.3,zorder=8)
    for i,goal in enumerate(GOALS['v2'],1):
        ax.add_patch(Circle(goal,.5,facecolor='#f3c450',alpha=.5,zorder=4))
        ax.scatter(*goal,marker='*',s=180,c='#e8b536',edgecolor='#956b12',lw=.6,zorder=9)
        ax.annotate(f'G{i}: {metrics["goals"][str(i)]}/40',goal,xytext=(0,18),textcoords='offset points',
                    ha='center',fontsize=10,fontweight='bold',color='#765718')
    title='Direct policy: random z + sigma' if mode.startswith('policy') else 'Supplement: random z, mu-only'
    ax.set_title(title+f'\nLeft {metrics["left"]} | right {metrics["right"]} | neither {metrics["neither"]}',fontsize=12,pad=10)
    ax.set_xlim(-14,14);ax.set_ylim(14,-14);ax.set_aspect('equal')
    ax.set_xticks([-8,0,8]);ax.set_yticks([-8,0,8]);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.suptitle('OptiQ AntMaze v2 | 256x3 | critic LR=5e-4 | 1.75M checkpoint\n'
             'Both directions explored; successful routes not diversified',fontsize=15,y=.97)
handles=[Line2D([],[],c=colors[0],lw=2,label='Left corridor entry'),Line2D([],[],c=colors[1],lw=2,label='Right corridor entry'),
         Line2D([],[],c=colors[3],lw=2,label='Neither entry'),Line2D([],[],marker='^',c='#253b49',ls='',label='Random start'),
         Line2D([],[],marker='x',c='#b35b53',ls='',label='Failed endpoint'),Line2D([],[],marker='o',c='#22925f',ls='',label='Successful endpoint')]
fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.065),frameon=False,fontsize=10)
fig.text(.5,.02,'40 random-start episodes per mode; all failures shown. Entry thresholds: x < -4 m / x > 4 m.\n'
         'Sparse + NovelD 0.01, T=0.01, seed0. No extra DACER noise in evaluation. Direction entry is not a successful route.',ha='center',fontsize=9,color='#4b5860')
fig.savefig(ROOT/'v2_routes.png',dpi=175,facecolor='white')
fig.savefig(ROOT/'v2_routes.pdf',facecolor='white')
(ROOT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
