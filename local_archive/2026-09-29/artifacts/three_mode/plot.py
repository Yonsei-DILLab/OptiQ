from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr
root=Path(__file__).parent;data=root/'results'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#2b79c2','#e88a30','#168d79'];grid=np.linspace(-.999,.999,1600)
target=sum(np.exp(-.5*((grid-c)/.1)**2)/(.1*np.sqrt(2*np.pi))/3 for c in [-.6,0,.6])
fig,axs=plt.subplots(2,3,figsize=(14,7.5),sharex=True,sharey=True)
for row,prefix in enumerate(['seed','fixed64_seed']):
 for col,step in enumerate([0,1000,5000]):
  ax=axs[row,col];snap=np.load(data/f'{prefix}0'/f'step{step}.npz')
  ax.hist(snap['samples'],bins=np.linspace(-1,1,129),density=True,alpha=.8,color=['#4285ef','#e88a30'][row],label='Policy: actual samples')
  ax.plot(grid,target,color='#25344a',lw=2,label='Target')
  ax.set_title(f'{step:,} updates');ax.set_xlim(-1,1);ax.set_ylim(0,2.6);ax.grid(alpha=.15)
  if col==0:ax.set_ylabel(['Resampled z | 2048 x 2048\nDensity','Fixed 64 z | 64 x 64\nDensity'][row]);ax.legend(fontsize=8)
  if row==1:ax.set_xlabel('Action')
fig.suptitle('OptiQ Direct GMM: how three target modes are covered',fontsize=18,y=.98)
fig.text(.5,.005,'Same original actor/loss, 5,000 updates, seed 0 shown. Fixed-z row evaluates its 64-component mixture. Raw histograms; no KDE.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.025,1,.95]);fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.7));steps=[0,100,500,1000,2000,5000]
fixed=[json.loads((data/f'fixed64_seed{s}'/'metrics.json').read_text()) for s in range(4)]
base=[json.loads((data/f'seed{s}'/'metrics.json').read_text()) for s in range(4)]
snap=np.load(data/'fixed64_seed0'/'step5000.npz');z=snap['z'][:,0];order=np.argsort(z)
for j in range(64):
 values=[float(np.tanh(np.load(data/'fixed64_seed0'/f'step{step}.npz')['mu'][j,0])) for step in steps]
 p=snap['probs'][j];color=colors[p.argmax()] if p.max()>=.8 else '#98a1b0'
 axs[0].plot(steps,values,color=color,alpha=.45,lw=1)
for c in [-.6,0,.6]:axs[0].axhline(c,color='#25344a',ls='--',alpha=.5)
axs[0].set(title='Each of the 64 fixed latents',xlabel='Update',ylabel='Conditional location tanh(mu)');axs[0].set_ylim(-1,1)
for j in range(64):
 mu=snap['mu'][j,0];sig=np.exp(snap['ls'][j,0]);u=np.arctanh(grid)
 density=np.exp(-.5*((u-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)*(1-grid**2))
 p=snap['probs'][j];color=colors[p.argmax()] if p.max()>=.8 else '#98a1b0'
 axs[1].plot(grid,density,color=color,lw=.8,alpha=.22)
axs[1].plot(grid,target,color='#25344a',lw=1.5,label='Target density')
axs[1].set(title='64 conditional densities at 5K',xlabel='Action',ylabel='Conditional density',ylim=(0,3.5));axs[1].legend(fontsize=8)
for offset,rows,name,color in [(-.18,base,'Resampled 2048', '#4285ef'),(.18,fixed,'Fixed 64','#e88a30')]:
 mass=np.array([r['rows'][-1]['mass'] for r in rows]);axs[2].bar(np.arange(3)+offset,mass.mean(0)*100,.34,yerr=mass.std(0,ddof=1)*100,label=name,color=color,capsize=3)
axs[2].axhline(100/3,color='#25344a',ls='--',label='Target ~33.3%');axs[2].set_xticks(range(3),['Left','Center','Right']);axs[2].set(title='Mode-basin mass: 4 seeds',ylabel='Sample share (%)',ylim=(0,60));axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.15)
fig.suptitle('Fixed latent bank: division across modes and remaining mismatch',fontsize=17)
fig.text(.5,.005,'Colored: conditional basin probability >= 80%. Gray: broad/ambiguous. Dashed lines are target centers; bars show mean +/- seed SD.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.03,1,.94]);fig.savefig(root/'fixed64_detail.png',dpi=170);plt.close(fig)
summary={}
for name,rows in [('resampled2048',base),('fixed64',fixed)]:
 summary[name]=dict(mean_mass=np.mean([r['rows'][-1]['mass'] for r in rows],0).tolist(),mean_tv=float(np.mean([r['rows'][-1]['tv'] for r in rows])),mean_specialist_fraction=float(np.mean([r['rows'][-1]['specialist_fraction'] for r in rows])),source_commit=rows[0]['source_commit'])
 if name=='fixed64':summary[name]['fresh_z_mean_tv']=float(np.mean([r['rows'][-1]['fresh_z_tv'] for r in rows]))
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
