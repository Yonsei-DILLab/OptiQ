from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path('tmp/reward-progress-worktree').resolve()))
from antmaze_experiments.progress_reward import maze_geometry
root=Path('artifacts/antmaze_euclidean_no_step/metric_reports/20260924T152830Z')
p=root/'raw/v3-policy-250k.npz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='8b5a964214a1cbaf9459a450e6cf7be7cde196e36aa72b4718562a91034fb1d7'
d=np.load(p);walls,goals,bounds=maze_geometry('v3')
colors={'left':'#2474B5','right':'#DF7E24','uncommitted':'#888888'}
fig,ax=plt.subplots(figsize=(7.8,7.8),layout='constrained')
for x1,y1,x2,y2 in walls:ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,facecolor='#e2e5e9',edgecolor='#b8bec7',lw=.6,zorder=0))
counts={k:0 for k in colors}
for xy,n in zip(d['xy'],d['lengths']):
 path=xy[:int(n)+1];left=(path[:,0]<-8).any();right=(path[:,0]>8).any()
 assert not(left and right)
 family='left' if left else 'right' if right else 'uncommitted';counts[family]+=1
 ax.plot(path[:,0],path[:,1],c=colors[family],lw=1,alpha=.57,zorder=2)
 ax.scatter(*path[-1],s=16,c=colors[family],marker='x',zorder=4)
assert counts==dict(left=12,right=23,uncommitted=5)
ax.scatter(goals[:,0],goals[:,1],c='#249150',marker='*',s=250,edgecolor='white',linewidth=.8,zorder=5)
ax.scatter(0,0,s=100,c='black',marker='o',edgecolor='white',zorder=6)
ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),xlabel='x (m)',ylabel='y (m)',aspect='equal')
ax.set_title('OptiQ v3 · 250k interactions · 40 rollouts\nEuclidean progress ×100 · no bonus / no step penalty\nDirect policy: random z + conditional sigma · fixed original start',fontsize=11,pad=12)
handles=[Line2D([0],[0],color=colors[k],lw=2,label=f'{label}: {counts[k]}/40') for k,label in [('left','Left (x < −8)'),('right','Right (x > 8)'),('uncommitted','Neither gate')]]
handles.extend([Line2D([0],[0],color='black',marker='o',ls='',label='Same start'),Line2D([0],[0],color='#249150',marker='*',ls='',markersize=12,label='Goals')])
ax.legend(handles=handles,loc='lower center',fontsize=9,framealpha=.94)
fig.suptitle('Both branches appear; goal success 0/40',fontweight='bold',fontsize=13)
fig.savefig(root/'v3_250k_trajectories.png',dpi=180)
print(root/'v3_250k_trajectories.png')
