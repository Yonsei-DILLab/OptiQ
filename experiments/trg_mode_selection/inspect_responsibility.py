"""Read-only inspection of existing TRG checkpoints; does not train or define modes."""
import argparse,json,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo));a.out.mkdir(parents=True,exist_ok=True)
import jax,jax.numpy as jnp,numpy as np
from flax import serialization
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.box_gaussian import component_log_prob
from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import connected_components
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
probe=np.load(a.data/'landscape_probe_batch.npz')['observations'];bs=len(probe);N=64
records=[];panels=[]
for ck in sorted(a.data.glob('actor_state_*.msgpack'),key=lambda f:int(f.stem.split('_')[-1])):
 step=int(ck.stem.split('_')[-1]);raw=serialization.msgpack_restore(ck.read_bytes());params=jax.tree_util.tree_map(jnp.asarray,raw['params'])
 dim=params['mu']['bias'].shape[0];actor=SemiImplicitActor(dim,(256,256),-5.,-1.,-1.)
 key,zk,pk=jax.random.split(jax.random.PRNGKey(919),3)
 z=jax.random.normal(zk,(bs,N,dim));obs=jnp.broadcast_to(jnp.asarray(probe)[:,None,:],(bs,N,probe.shape[1]))
 mu,ls=actor.apply({'params':params},obs.reshape(bs*N,-1),z.reshape(bs*N,dim));mu=mu.reshape(bs,N,dim);ls=ls.reshape(bs,N,dim)
 proposal=ConditionalGaussianProposal(mu,ls,float(np.exp(-5)))
 actions,_,_=proposal.sample(pk,1,'exact');ell=component_log_prob(actions,mu,ls)
 gamma=np.asarray(jax.nn.softmax(ell,axis=1));H=np.sqrt(gamma);similarity=H.transpose(0,2,1)@H
 counts={str(th):[int(connected_components(s>=th,directed=False,return_labels=False)) for s in similarity] for th in (.8,.9,.95,.99)}
 entropy=-(gamma*np.log(np.maximum(gamma,1e-30))).sum(1)
 row=0;g=gamma[row]
 ro=leaves_list(linkage(pdist(np.sqrt(g)),method='average',optimal_ordering=True))
 co=leaves_list(linkage(pdist(np.sqrt(g.T)),method='average',optimal_ordering=True))
 rec=dict(step=step,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),states=bs,N=N,M=N,
  normalized_responsibility_entropy_mean=float(entropy.mean()/np.log(N)),posterior_max_mean=float(gamma.max(1).mean()),
  graph_group_count_mean={th:float(np.mean(v)) for th,v in counts.items()},graph_group_count_range={th:[min(v),max(v)] for th,v in counts.items()},
  caveat='Graph groups are responsibility-pattern clusters, not ground-truth Q modes. No threshold is selected for training.')
 records.append(rec);panels.append((step,g,g[ro][:,co],similarity[row]))
 np.savez_compressed(a.out/f'responsibility_{step}.npz',gamma=gamma,actions=np.asarray(actions),mu=np.asarray(mu),log_std=np.asarray(ls),observations=probe,z=np.asarray(z),similarity=similarity,sorted_rows=ro,sorted_columns=co)
fig,axs=plt.subplots(len(panels),3,figsize=(13,3.5*len(panels)),squeeze=False)
for row,(step,raw,sortedg,sim) in enumerate(panels):
 for col,(x,title) in enumerate(((raw,'Responsibility: original order'),(sortedg,'Responsibility: reordered only'),(sim,'Candidate posterior similarity'))):
  im=axs[row,col].imshow(x,aspect='auto',origin='lower',vmin=0,vmax=1 if col==2 else max(float(raw.max()),1/N),cmap='viridis')
  axs[row,col].set_title(f'{step:,} steps | {title}',fontsize=10);axs[row,col].set_xlabel('Teacher candidate');axs[row,col].set_ylabel('Student latent' if col<2 else 'Teacher candidate');fig.colorbar(im,ax=axs[row,col])
fig.suptitle('Existing TRG Humanoid, seed 0 | N=M=64 | fixed probe state #0',fontsize=13);fig.tight_layout();fig.savefig(a.out/'responsibility_heatmaps.png',dpi=170)
(a.out/'SUMMARY.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
