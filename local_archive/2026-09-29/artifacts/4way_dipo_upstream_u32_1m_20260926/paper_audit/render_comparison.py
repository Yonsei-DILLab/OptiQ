from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
out=root/'paper_audit'
p=np.load(root/'evaluations/001001472_probe.npz')
r=np.load(root/'evaluations/001001472_policy.npz')
x,y,a=p['x'],p['y'],p['actions']
goals=np.array([[5,0],[-5,0],[0,5],[0,-5]])
fig,axes=plt.subplots(1,2,figsize=(12.5,5.8),layout='constrained')
axis=np.linspace(-7,7,281);xx,yy=np.meshgrid(axis,axis)
proximity=sum(np.exp(-((xx-gx)**2+(yy-gy)**2)/(2*1.7**2)) for gx,gy in goals)
for ax in axes:
 ax.contour(xx,yy,proximity,levels=12,cmap='viridis',linewidths=.6,alpha=.5)
 ax.scatter(goals[:,0],goals[:,1],c='limegreen',s=95,marker='s',edgecolors='darkgreen',zorder=5)
 for name,(gx,gy) in zip(['E','W','N','S'],goals):ax.text(gx+.25,gy+.2,name,weight='bold')
 ax.scatter(0,0,color='black',s=24,zorder=6)
 ax.set(xlim=(-7,7),ylim=(-7,7),aspect='equal',xlabel='x',ylabel='y')
coords=np.arange(-3,4);gx,gy=np.meshgrid(coords,coords)
ix=np.array([np.argmin(abs(x-v)) for v in gx.ravel()]);iy=np.array([np.argmin(abs(y-v)) for v in gy.ravel()])
# One stored independent action sample per grid state, matching paper G.2.1.
a49=a[iy,ix,0]
axes[0].quiver(gx.ravel(),gy.ravel(),a49[:,0],a49[:,1],angles='xy',scale_units='xy',scale=1/1.5,color='crimson',width=.0055,headwidth=3.5)
axes[0].set_title('49 different states: paper-style action field\nOne saved policy draw per state; arrows enlarged 1.5x',fontsize=11)
for xy,length in zip(r['xy'],r['lengths']):
 track=xy[:int(length)+1]
 axes[1].plot(track[:,0],track[:,1],color='#d400c8',lw=2.0,alpha=.12)
axes[1].set_title('Same fixed origin: actual 1,024 rollouts\nE 0 | W 0 | N 0 | S 1,024; success 100%',fontsize=11)
fig.suptitle('DIPO 4-Way | same checkpoint at 1,001,472 transitions',fontsize=14,weight='bold')
fig.supxlabel('Fresh Gaussian initial noise; reverse-step noise OFF (upstream evaluation).\nBackground contours illustrate goal proximity, not learned policy density.',fontsize=9)
fig.savefig(out/'paper_grid_vs_origin_rollouts.png',dpi=190)
plt.close(fig)
origin=a[np.argmin(abs(y)),np.argmin(abs(x))]
report={'source_commit':'5f4b809d0e95899eef2435dd80cac2da48058931','upstream_commit':'c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc','step':1001472,'learner_updates':15520,'origin_8_probe_action_mean':origin.mean(0).tolist(),'origin_8_probe_action_std':origin.std(0).tolist(),'goal_counts_E_W_N_S':np.bincount(r['goal_ids'].astype(int),minlength=4).tolist(),'figure_protocol':'49 integer states in [-3,3]^2, first saved draw per state; no new inference or training','input_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'evaluations/001001472_probe.npz',root/'evaluations/001001472_policy.npz',root/'config.json']}}
(out/'probe_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
