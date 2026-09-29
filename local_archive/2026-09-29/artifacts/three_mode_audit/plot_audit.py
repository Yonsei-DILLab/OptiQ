from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;grid=np.linspace(-1,1,1200);target=sum(np.exp(-.5*((grid-c)/.1)**2)/(.1*np.sqrt(2*np.pi))/3 for c in [-.6,0,.6])
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True,sharey=True)
for i,variant in enumerate(['original','init1']):
 for seed in [0,1]:
  ax=axes[i,seed];folder=p/f'{variant}_seed{seed}';d=np.load(folder/'step100000.npz');m=json.loads((folder/'metrics.json').read_text())['rows'][-1]
  ax.hist(d['samples'],bins=np.linspace(-1,1,129),density=True,color=['#e58c35','#1f9c94'][i],alpha=.75,label='Fixed 64-mixture samples')
  ax.plot(grid,target,color='#26334b',lw=2,label='Target: 3 modes')
  ax.set_title(f"Seed {seed} | TV {m['tv']:.3f} | specialist latents {m['specialist_fraction']*100:.1f}%")
  ax.set_xlim(-1,1);ax.set_ylim(0,2.25);ax.grid(alpha=.15)
  if seed==0:ax.set_ylabel(['Original mean init (1e-4)\nDensity','Larger mean init (1.0)\nDensity'][i])
  if i==1:ax.set_xlabel('Action')
axes[0,0].legend(fontsize=9)
fig.suptitle('Direct GMM | fixed z64 | N=M=64 | T=1 | batch=32 | 100,000 updates',fontsize=16)
fig.text(.5,.012,'Only mean-head initialization differs between rows. Same z bank per seed, sigma init=0.5, Adam, network and zero latent skip.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(p/'audit_comparison.png',dpi=170)
for variant in ['original','init1']:
 d=np.load(p/f'{variant}_seed0'/'step0.npz');print(variant,'initial mu std',d['mu'].std())
