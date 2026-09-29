from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.colors import LogNorm
root=Path(__file__).resolve().parent
summary=json.loads((root/'original-training-audit.json').read_text())
with np.load(root/'original-training-heatmaps.npz') as z:
 data={k:z[k] for k in z.files}
methods=['optiq','sac','mfpo','meow']; names=['OptiQ','SAC','MFPO','MEOW']
vmax=max(data[m+'_'+w].max() for m in methods for w in ['first100k','last100k'])
fig,axes=plt.subplots(2,4,figsize=(16,9.8),layout='constrained')
cm=plt.get_cmap('magma').copy();cm.set_bad('#f2f2f2')
for col,(method,name) in enumerate(zip(methods,names)):
 g=summary[method]['geometry']
 for row,w in enumerate(['first100k','last100k']):
  ax=axes[row,col];h=data[method+'_'+w]
  im=ax.imshow(np.ma.masked_where(h.T==0,h.T),origin='lower',extent=(-18,18,-18,18),cmap=cm,norm=LogNorm(vmin=1,vmax=vmax),interpolation='nearest')
  for x,y in g['walls']:ax.add_patch(Rectangle((x-2,y-2),4,4,color='#4b5054',zorder=3))
  for i,(x,y) in enumerate(g['goals'],1):
   ax.add_patch(Circle((x,y),.5,color='#37bd72',zorder=4));ax.text(x,y+1,'G'+str(i),ha='center',fontsize=8,color='#126237',zorder=5)
  ax.scatter(0,0,marker='*',color='#22b9ec',s=60,zorder=6)
  ax.set(xlim=(-18,18),ylim=(-18,18),aspect='equal',xlabel='x',ylabel='y')
  window='First 100k steps' if row==0 else 'Last 100k steps'
  counts=summary[method]['windows'][w]['successful_episodes_per_goal']
  ax.set_title(f'{name} · {window}\nTraining success: G1 {counts.get("1",0)}, G2 {counts.get("2",0)}',fontsize=10)
fig.suptitle('AntMaze v3 · original training replay · seed 0\nWhere the policy actually visited during learning',fontsize=15)
fig.colorbar(im,ax=axes,label='Visits per 0.5 × 0.5 cell (log scale)',shrink=.7)
for ext in ['png','pdf']:fig.savefig(root/f'training-coverage-early-late.{ext}',dpi=170)
plt.close(fig)
records=json.loads((root/'original-noveld-metrics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),layout='constrained')
for m,n in zip(methods,names):
 rs=records[m];steps=np.array([r['env_steps'] for r in rs]);b=np.array([r['intrinsic_mean'] for r in rs]);er=np.abs([r['env_reward_mean'] for r in rs])
 # Fixed 20-log trailing mean for readability; raw recorded batch statistics are retained.
 def smooth(y):return np.convolve(y,np.ones(20)/20,mode='valid')
 axes[0].plot(steps[19:]/1e6,smooth(b),label=n)
 axes[1].plot(steps[19:]/1e6,smooth(er),label=n)
axes[0].set(ylabel='Mean intrinsic bonus in replay batch',title='NovelD bonus actually added')
axes[1].set(ylabel='Absolute mean environment reward in replay batch',title='Dense nearest-goal reward magnitude')
for ax in axes:ax.set(xlabel='Environment interactions (million)',yscale='log');ax.grid(alpha=.2);ax.legend(frameon=False)
fig.suptitle('AntMaze v3 · recorded training reward components\nDifferent vertical scales; magnitude alone does not measure causal influence',fontsize=12)
for ext in ['png','pdf']:fig.savefig(root/f'noveld-reward-scale.{ext}',dpi=170)
plt.close(fig)
