"""Quick 3-mode coverage illustration using unchanged heejoon Direct GMM."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from experiments.gmm_gradient_interference.core import initialize,block,draw,heads,basin_prob,problems
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
state,key=initialize(a.seed);train=block(2048,2048)
steps=[0,100,500,1000,2000,5000];rows=[];snaps={};t0=time.time()
edges=np.linspace(-1,1,257);target=np.diff(problems.analytic_cdf(edges,problems.schedule('prefix',1)))
z=jax.random.normal(jax.random.PRNGKey(90000+a.seed),(2048,1))
for i,step in enumerate(steps):
 if i:state,key,loss=train(state,key,step-steps[i-1]);jax.block_until_ready(state.params)
 samples=np.asarray(draw(state.params,jax.random.PRNGKey(80000+a.seed),32768));hist=np.histogram(samples,edges)[0]/len(samples)
 mu,ls=heads(state.params,z);probs=np.asarray(basin_prob(mu,ls));mass=np.histogram(samples,[-1,-.3,.3,1])[0]/len(samples)
 row=dict(step=step,mass=mass.tolist(),tv=float(abs(hist-target).sum()/2),specialist_fraction=float((probs.max(1)>=.8).mean()),seconds=time.time()-t0)
 rows.append(row);snaps[str(step)]=dict(samples=samples,mu=np.asarray(mu),ls=np.asarray(ls),probs=probs)
 (out/'metrics.json').write_text(json.dumps(dict(source_commit=sha,n=2048,m=2048,seed=a.seed,rows=rows),indent=2))
 np.savez_compressed(out/f'step{step}.npz',**snaps[str(step)])
 print(json.dumps(dict(seed=a.seed,**row)),flush=True)
 if a.seed==0:
  fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
  grid=np.linspace(-1,1,1200);pdf=np.exp(problems.analytic_q(grid,problems.schedule('prefix',1))/.25)
  for ax,s in zip(axs.flat,steps):
   ax.plot(grid,pdf,color='#202a44',lw=2,label='Target: 3 modes')
   if str(s) in snaps:ax.hist(snaps[str(s)]['samples'],bins=edges,density=True,color='#4285ef',alpha=.75,label='Policy samples')
   ax.set_title(f'{s:,} updates'+('' if str(s) in snaps else ' (pending)'));ax.set_xlim(-1,1);ax.set_ylim(0,2.8);ax.grid(alpha=.15)
  axs[0,0].legend(fontsize=8);fig.supxlabel('Action');fig.supylabel('Density');fig.suptitle('OptiQ Direct GMM | 3-mode coverage | N = M = 2048 | seed 0',fontsize=15)
  fig.tight_layout();fig.savefig(out/'progress.png',dpi=150);plt.close(fig)
print('COMPLETE',flush=True)
