"""Actual RL Q-weighted row assignments; no mode labels or training updates."""
import argparse,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo));a.out.mkdir(parents=True,exist_ok=True)
import jax,jax.numpy as jnp,numpy as np
from flax import serialization
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.box_gaussian import component_log_prob
from models.critic import VectorCritic
from models.utils import activation_fn
from scipy.special import logsumexp
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cfg=json.loads((a.data/'resolved_config.json').read_text());ac=cfg['alg']['actor'];cc=cfg['alg']['critic'];oc=cfg['alg']['optimizer']
assert ac['distillation_loss']=='direct_gmm_nll' and ac['density_correction'] and not ac['adaptive_density_beta']
assert ac['source_q_eval']=='mean' and ac['teacher_distribution']=='conditional_mixture'
assert cc['n_critics']==1 and cc['n_atoms']==1 and not oc['bn']
tau=ac['temperature'];beta=ac['density_beta'];probe=np.load(a.data/'landscape_probe_batch.npz');observations=probe['observations'];probe_step=int(probe['source_step'])
steps=(50000,650000);records=[];groups={};inputs={}
critic=VectorCritic(net_arch=tuple(cc['hs']),activation_fn=activation_fn[cc['activation']],batch_norm_momentum=oc['bn_momentum'],bn_warmup=oc['bn_warmup'],use_batch_norm=oc['bn'],batch_norm_mode=oc['bn_mode'],use_layer_norm=cc['use_layer_norm'],dropout_rate=cc['dropout_rate'],n_critics=cc['n_critics'],n_atoms=cc['n_atoms'])
critic_apply=jax.jit(lambda p,b,o,x:critic.apply({'params':p,'batch_stats':b},o,x,train=False))

def inspect(step,n,m,obs,tag):
 actorfile=a.data/f'actor_state_{step}.msgpack';qfile=a.data/f'critic_state_{step}.msgpack'
 actorstate=serialization.msgpack_restore(actorfile.read_bytes());qstate=serialization.msgpack_restore(qfile.read_bytes())
 params=jax.tree_util.tree_map(jnp.asarray,actorstate['params']);qparams=jax.tree_util.tree_map(jnp.asarray,qstate['params']);qstats=jax.tree_util.tree_map(jnp.asarray,qstate.get('batch_stats',{}))
 dim=params['mu']['bias'].shape[0];bs=len(obs)
 actor=SemiImplicitActor(dim,tuple(ac['hidden_dims']),ac['log_std_min'],ac['log_std_max'],ac['initial_log_std'],ac['mean_output_init_scale'],ac['log_std_output_init_scale'],ac.get('mean_latent_skip_scale',0.))
 _,zk,pk=jax.random.split(jax.random.PRNGKey(919),3)
 z=jax.random.normal(zk,(bs,n,dim));obsj=jnp.asarray(obs);ob=jnp.broadcast_to(obsj[:,None,:],(bs,n,obs.shape[-1]))
 mu,ls=actor.apply({'params':params},ob.reshape(bs*n,-1),z.reshape(bs*n,dim));mu=mu.reshape(bs,n,dim);ls=ls.reshape(bs,n,dim)
 proposal=ConditionalGaussianProposal(mu,ls,ac['teacher_std_floor']);b,u,_=proposal.sample(pk,m//n,'exact')
 ell=np.asarray(component_log_prob(b,mu,ls));lq=np.asarray(proposal.log_prob(u))
 # Same LIVE critic and scalar expectation as the actor's training teacher.
 ob=jnp.broadcast_to(obsj[:,None,:],(bs,m,obs.shape[-1])).reshape(bs*m,-1);flatb=b.reshape(bs*m,dim)
 qs=[]
 for start in range(0,len(flatb),2048):
  q=critic_apply(qparams,qstats,ob[start:start+2048],flatb[start:start+2048]);assert q.shape[0]==1 and q.shape[-1]==1
  qs.append(np.asarray(q[0,:,0]))
 Q=np.concatenate(qs).reshape(bs,m)
 # Reproduce float32 teacher softmax; use float64 log space for diagnostic
 # row normalization so underused rows do not produce 0/0 from underflow.
 logits=Q/tau-beta*lq;logw=np.asarray(jax.nn.log_softmax(jnp.asarray(logits),axis=-1));w=np.exp(logw)
 eg=ell.astype(np.float64);lg=eg-logsumexp(eg,axis=1,keepdims=True)
 logJ=lg+logw[:,None,:];loga=logsumexp(logJ,axis=-1);R=np.exp(logJ-loga[:,:,None]);alpha=np.exp(loga)
 np.testing.assert_allclose(R.sum(-1),1.,atol=1e-10);np.testing.assert_allclose(np.exp(logJ).sum(1),w,atol=3e-7,rtol=2e-6)
 np.testing.assert_allclose(alpha.sum(-1),1.,atol=3e-6)
 assert np.isfinite(Q).all() and np.isfinite(R).all() and np.isfinite(loga).all()
 # Independent quotient check for representable nonzero rows.
 joint=np.exp(logJ);mass=joint.sum(-1);valid=mass>np.finfo(float).tiny
 quot=joint[valid]/mass[valid,None]
 np.testing.assert_allclose(R[valid],quot,atol=1e-10,rtol=1e-9)
 # Within each state, recover teacher mass from row conditionals + usage.
 np.testing.assert_allclose((alpha[:,:,None]*R).sum(1),w,atol=3e-7,rtol=2e-6)
 data=dict(R=R.astype(np.float32),log_alpha=loga.astype(np.float32),alpha=alpha.astype(np.float32),gamma=np.exp(lg).astype(np.float32),weights=w,Q=Q,log_q=lq,mu=np.asarray(mu),log_sigma=np.asarray(ls),candidates=np.asarray(b),z=np.asarray(z),observations=obs)
 np.savez_compressed(a.out/f'{tag}_{step}.npz',**data)
 inputs[actorfile.name]=hashlib.sha256(actorfile.read_bytes()).hexdigest();inputs[qfile.name]=hashlib.sha256(qfile.read_bytes()).hexdigest()
 stats=dict(tag=tag,step=step,N=n,M=m,states=bs,teacher_ess_mean=float((1/(w*w).sum(-1)).mean()),teacher_wmax_mean=float(w.max(-1).mean()),row_usage_ess_mean=float((1/(alpha*alpha).sum(-1)).mean()),row_norm_max_error=float(np.abs(R.sum(-1)-1).max()),state0_teacher_ess=float(1/(w[0]**2).sum()),state0_teacher_wmax=float(w[0].max()),state0_underused_rows=int((alpha[0]<.1/n).sum()))
 records.append(stats);print(json.dumps(stats),flush=True)
 return data

for step in steps:
 groups[('actual',step)]=inspect(step,64,64,observations,'actual_64x64')
 groups[('dense',step)]=inspect(step,256,16384,observations[:1],'dense_256x16384')

# One fixed projection direction per probe state, shared across checkpoints.
# For state0 it also spans both diagnostic sample budgets. PCA only orders
# rows/columns and has no access to Q, weights, mode labels or heatmap contrast.
def axis_for(s):
 x=np.concatenate([d[key][s] for (kind,step),d in groups.items() if s<len(d['observations']) for key in ('mu','candidates')],axis=0).astype(np.float64)
 center=x.mean(0);_,sv,vt=np.linalg.svd(x-center,full_matrices=False);v=vt[0]
 if v[np.argmax(np.abs(v))]<0:v=-v
 return v,float(sv[0]**2/(sv@sv))
axes={s:axis_for(s) for s in range(8)}

def order(d,s,method):
 if method=='original':return np.arange(d['mu'].shape[1]),np.arange(d['candidates'].shape[1])
 v=np.eye(d['mu'].shape[-1])[0] if method=='coordinate0' else axes[s][0]
 return np.argsort(d['mu'][s]@v,kind='stable'),np.argsort(d['candidates'][s]@v,kind='stable')

cm=plt.get_cmap('magma').copy();cm.set_bad('black');norm=LogNorm(vmin=1e-5,vmax=1)
plt.rcParams.update({'font.size':11,'axes.titlesize':11,'savefig.bbox':'tight'})

def draw(ax,d,s,method,block):
 rows,cols=order(d,s,method);mat=d['R'][s][rows][:,cols]
 if block>1:mat=mat.reshape(len(rows),-1,block).sum(2)
 im=ax.imshow(np.ma.masked_equal(mat,0),origin='lower',aspect='auto',interpolation='nearest',cmap=cm,norm=norm)
 ax.set_xlabel('Candidate index' if block==1 else f'Candidate-index block ({block} columns summed)');ax.set_ylabel('Student row')
 return im

for kind,tag,block in [('actual','actual_64x64',1),('dense','dense_256x16384',32)]:
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for row,step in enumerate(steps):
  d=groups[(kind,step)]
  for col,method in enumerate(('original','coordinate0','pca')):
   im=draw(axs[row,col],d,0,method,block);title={'original':'Original order','coordinate0':'Sorted by action coordinate 0','pca':'Sorted by common action PC1'}[method]
   axs[row,col].set_title(f'{step:,} steps | {title}')
 fig.colorbar(im,ax=axs.ravel().tolist(),label='Conditional row mass R (log scale)',shrink=.82)
 n,m=groups[(kind,steps[0])]['R'].shape[1:]
 budget_label='training-size sampling' if kind=='actual' else 'DENSE DIAGNOSTIC ONLY: trained at 64x64'
 fig.suptitle(f'TRG Humanoid seed 0 | N={n}, M={m:,} | fixed probe state #0\n{budget_label} | R = (weight x responsibility) / row mass | no mode labels',fontsize=13)
 fig.savefig(a.out/f'{tag}.png',dpi=180);fig.savefig(a.out/f'{tag}.pdf');plt.close(fig)

# Additional states are the first eight probes, not selected by appearance.
for step in steps:
 fig,axs=plt.subplots(4,4,figsize=(15,13),layout='constrained');d=groups[('actual',step)]
 for s in range(8):
  row=s//2;pair=s%2
  for j,method in enumerate(('original','pca')):
   ax=axs[row,pair*2+j];im=draw(ax,d,s,method,1);ax.set_title(f'State #{s} | '+('Original' if j==0 else 'Common PC1 order'))
 fig.colorbar(im,ax=axs.ravel().tolist(),label='Conditional row mass R (log scale)',shrink=.6)
 fig.suptitle(f'TRG Humanoid seed 0 | {step:,} steps | actual N=M=64\nFirst eight fixed probes; no selection by block visibility',fontsize=14)
 fig.savefig(a.out/f'first8_states_{step}.png',dpi=155);plt.close(fig)

summary=dict(records=records,input_hashes=inputs,probe_source_step=probe_step,temperature=tau,beta=beta,training_updates=0,
 Q_source='saved live single critic, not target critic',row_normalization='float64 log-domain diagnostic normalization; float32 Q/proposal and teacher softmax',
 projection={str(s):dict(vector=v.tolist(),explained_variance_fraction=r) for s,(v,r) in axes.items()},
 actual_sampling='N=M64, same actor family/proposal rule as training; fresh diagnostic samples at fixed saved replay states',
 dense_sampling='N256 M16384 diagnostic resampling ONLY; checkpoint was trained with N=M64',
 plot='shared LogNorm[1e-5,1], R rows normalized before optional contiguous column pooling; no mode labels',
 caveat='Row normalization hides component total usage alpha; a bright row does not imply a frequently used actor component.')
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
