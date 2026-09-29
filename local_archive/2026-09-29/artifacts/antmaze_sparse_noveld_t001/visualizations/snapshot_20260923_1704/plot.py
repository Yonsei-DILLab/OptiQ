"""Latest saved sparse/T=.01 evaluations; no learning or new rollouts."""
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

COLORS = ['#2082b4', '#e48a37', '#8b62af', '#8d969d']
TASKS = ['v1', 'v2', 'v3', 'v4']


def read(task, mode):
    run = ROOT / 'raw' / f'{task}-optiq-s0'
    config = json.loads((run/'config.json').read_text())
    assert config['source_commit'] == '8bb0c50a1356dc082106393f161ff3fc3d2de0af'
    assert config['temperature'] == .01 and config['noveld_coefficient'] == .01
    assert config['reward_profile'] == 'sparse'
    checkpoints = [p for p in sorted((run/'evaluations').iterdir())
                   if all((p/f'{m}-natural'/'summary.json').exists() for m in ['policy','native'])]
    folder = checkpoints[-1] / f'{mode}-natural'
    summary = json.loads((folder/'summary.json').read_text())
    path = folder/'rollouts.npz'
    with np.load(path) as f:
        d = {k:f[k].copy() for k in f.files}
    xy, lengths, goals = d['xy'], d['lengths'], d['goals']
    assert len(xy) == summary['episodes'] == 40 and not bool(d['fixed'])
    assert len(np.unique(d['initial_full_state'],axis=0)) == 40
    expected = np.where(goals > 0,10.,0.)
    if task == 'v2': expected = np.where(goals == 1,20.,expected)
    assert np.allclose(d['returns'],expected)
    for line,n in zip(xy,lengths):
        assert np.isfinite(line[:n+1]).all() and np.isnan(line[n+1:]).all()
    distance = np.linalg.norm(xy[:,:,None,:]-np.array(GOALS[task])[None,None],axis=-1)
    reached = np.nanmin(distance,axis=(1,2)) <= .50002
    assert np.array_equal(reached,goals>0)
    if task in ('v1','v4'):
        a = np.any((xy[:,:,0]<-2)&(xy[:,:,1]<-2),axis=1)
        b = np.any((xy[:,:,0]<-2)&(xy[:,:,1]>2),axis=1)
        labels=['upper','lower']
        rule='Upper/lower entry: x<-2 and y<-2 / y>2. Entries can overlap.'
    else:
        a = np.any(xy[:,:,0]<-4,axis=1)
        b = np.any(xy[:,:,0]>4,axis=1)
        labels=['left','right']
        rule='Left/right entry: any x<-4 / x>4. Entries can overlap.'
    category=np.where(a&b,2,np.where(a,0,np.where(b,1,3)))
    points=xy.reshape(-1,2);points=points[np.isfinite(points).all(axis=1)]
    progress=json.loads((run/'progress.json').read_text())
    metrics=dict(step=int(d['env_steps']),current_step=progress['step'],budget=config['steps'],
        training_successes=progress['successes'],episodes=40,success_count=int(reached.sum()),
        entries={labels[0]:int(a.sum()),labels[1]:int(b.sum()),'both':int((a&b).sum()),
                 'neither':int((~a&~b).sum())},route_definition=rule,
        nearest_goals_m=np.nanmin(distance,axis=(0,1)).tolist(),
        max_distance_from_start_m=np.nanmax(np.linalg.norm(xy-xy[:,:1],axis=-1)).item(),
        xy_min=np.nanmin(xy,axis=(0,1)).tolist(),xy_max=np.nanmax(xy,axis=(0,1)).tolist(),
        evaluation_visited_bins_0p5m=len(np.unique(np.floor(points/.5),axis=0)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return d,metrics,category,labels


def background(ax,task):
    maze,rr,cc=geometry(task)
    for i,row in enumerate(maze):
        for j,cell in enumerate(row):
            if cell==1:
                ax.add_patch(Rectangle(((j-cc)*4-2,(i-rr)*4-2),4,4,
                    facecolor='#e4e9ed',edgecolor='#c1c9cf',lw=.5,zorder=0))
    ax.add_patch(Rectangle((-2,-2),4,4,facecolor='#dfeee4',lw=0,zorder=1))
    for i,goal in enumerate(GOALS[task],1):
        ax.add_patch(Circle(goal,.5,color='#e5bd48',alpha=.4,zorder=5))
        ax.scatter(*goal,marker='*',s=160,c='#dcaa26',edgecolor='#826411',lw=.6,zorder=7)
        ax.annotate(f'G{i}',goal,xytext=(8,-14),textcoords='offset points',fontsize=9,color='#826411')
    ax.set_xlim(-cc*4-2,(len(maze[0])-1-cc)*4+2)
    ax.set_ylim((len(maze)-1-rr)*4+2,-rr*4-2)
    ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.grid(alpha=.12)


def plot(mode):
    fig,axes=plt.subplots(2,2,figsize=(13,12))
    fig.subplots_adjust(left=.065,right=.97,bottom=.20,top=.87,hspace=.40,wspace=.20)
    result={}
    for ax,task in zip(axes.flat,TASKS):
        d,m,categories,labels=read(task,mode);result[task]=m
        background(ax,task)
        for xy,n,c in zip(d['xy'],d['lengths'],categories):
            line=xy[:n+1]
            ax.plot(line[:,0],line[:,1],color=COLORS[c],lw=.9,alpha=.63,zorder=3)
            ax.scatter(*line[-1],c='#c13e47',marker='x',s=16,lw=.7,alpha=.65,zorder=5)
        ax.scatter(d['xy'][:,0,0],d['xy'][:,0,1],c='#203448',marker='^',s=17,
                   edgecolor='white',lw=.3,zorder=6)
        ax.set_title(f"{task.upper()} | evaluation {m['step']/1e6:.2f}M | success {m['success_count']}/40",
                     fontsize=12,pad=10)
        text=f"{labels[0].capitalize()} entry {m['entries'][labels[0]]}  |  {labels[1]} entry {m['entries'][labels[1]]}  |  neither {m['entries']['neither']}"
        ax.text(.5,-.19,text,transform=ax.transAxes,ha='center',fontsize=10)
    caption='Direct policy: random z + conditional sigma' if mode=='policy' else 'Supplement: random z, mu-only'
    fig.suptitle('OptiQ | sparse reward + NovelD 0.01 | T = 0.01 | seed 0\n'+caption,
                 fontsize=15,y=.965,linespacing=1.5)
    handles=[Line2D([],[],color=COLORS[0],label='Upper / left entry'),
             Line2D([],[],color=COLORS[1],label='Lower / right entry'),
             Line2D([],[],color=COLORS[2],label='Both'),
             Line2D([],[],color=COLORS[3],label='Neither'),
             Line2D([],[],marker='^',ls='',color='#203448',label='Random start'),
             Line2D([],[],marker='x',ls='',color='#c13e47',label='Endpoint')]
    fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False,fontsize=10)
    fig.text(.5,.015,'40 random-start rollouts per maze; all failures included. No extra DACER noise or NovelD in evaluation.\n'
             'Different checkpoint steps. Corridor entry is not a successful route or proof of state-conditional multimodality.',
             ha='center',fontsize=9,color='#4c5760')
    fig.savefig(ROOT/f'trajectories_{mode}.png',dpi=170,facecolor='white')
    fig.savefig(ROOT/f'trajectories_{mode}.pdf',facecolor='white');plt.close(fig)
    return result


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    out={'source_commit':'8bb0c50a1356dc082106393f161ff3fc3d2de0af',
         'primary':plot('policy'),'supplementary':plot('native')}
    for task in TASKS:
        a,*_=read(task,'policy');b,*_=read(task,'native')
        assert np.array_equal(a['initial_full_state'],b['initial_full_state'])
    (ROOT/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
