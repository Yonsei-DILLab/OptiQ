"""Direct GMM on pinned GMM40: fixed64 versus freshly sampled Gaussian latent."""
import argparse,hashlib,json,time,subprocess,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax.training.train_state import TrainState
from flax import serialization
from experiments.gmm_gradient_interference import bootstrap
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.distillation import direct_gmm_nll

P=Path(__file__).parent
META=json.loads((P/'target.json').read_text());MEANS=np.asarray(META['means']);STD=np.asarray(META['std'])
N=M=64;B=32;D=2;SCALE=40.;EVAL=32768

def sample_target(count,seed):
 rng=np.random.default_rng(seed);parts=[];total=0
 while total<count:
  idx=rng.integers(40,size=max(1024,count-total));x=MEANS[idx]+STD[idx,None]*rng.normal(size=(len(idx),2));x=x[np.all(abs(x)<40,axis=1)];parts.append(x);total+=len(x)
 return np.concatenate(parts)[:count].astype(np.float32)

def assign(x):
 dist=cdist(x,MEANS,'sqeuclidean')/STD[None,:]**2;labels=dist.argmin(1);near=dist[np.arange(len(x)),labels]<=9
 return labels,near,np.bincount(labels[near],minlength=40)

def mmd2(x,y):
 x=x[:2048].astype(float);y=y[:2048].astype(float);xx=cdist(x,x,'sqeuclidean');yy=cdist(y,y,'sqeuclidean');xy=cdist(x,y,'sqeuclidean')
 return float(np.mean([(np.exp(-xx/(2*h*h)).sum()-len(x))/(len(x)*(len(x)-1))+(np.exp(-yy/(2*h*h)).sum()-len(y))/(len(y)*(len(y)-1))-2*np.exp(-xy/(2*h*h)).mean() for h in [1,2,5,10,20]]))

def metrics(x,reference):
 assert x.shape==(EVAL,2) and np.isfinite(x).all()
 _,near,c=assign(x);_,_,rc=assign(reference);threshold=np.maximum(10,.1*rc*len(x)/len(reference))
 return dict(coverage=int((c>=threshold).sum()),near_fraction=float(near.mean()),mode_mass_tv=float(.5*np.abs(c/max(c.sum(),1)-rc/rc.sum()).sum()),mmd2=mmd2(x,reference),counts=c.tolist(),coverage_threshold=threshold.tolist())

@jax.jit
def qvalue(x):
 diff=(x[...,None,:]-jnp.asarray(MEANS))/jnp.asarray(STD)[:,None]
 return jsp.special.logsumexp(-.5*jnp.square(diff).sum(-1)-2*jnp.log(jnp.asarray(STD))-jnp.log(2*jnp.pi),axis=-1)-jnp.log(40.)

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['fixed','fresh'],required=True);p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=256);a=p.parse_args();B=a.batch
out=Path(a.out)/f'{a.mode}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
manifest=dict(source_commit=source,mode=a.mode,seed=a.seed,n=N,m=M,batch=B,temperature=1.,q='log p_GMM40',steps=100000,actor_hidden=[256,256],mean_output_init_scale=1.,sigma_init=.5,latent_skip=0.,learning_rate=3e-4,teacher_sigma_floor=.05,action_scale=SCALE,eval_samples=EVAL,primary_prior='uniform fixed 64 codes' if a.mode=='fixed' else 'standard Gaussian continuous latent',target_branch_commit='a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71',target_file_sha256=hashlib.sha256((P/'target.json').read_bytes()).hexdigest(),packages={k:importlib.metadata.version(k) for k in ['jax','jaxlib','flax','numpy','optax','scipy']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
model=SemiImplicitActor(D,(256,256),-5.,1.,float(np.log(.5)),mean_output_init_scale=1.)
params=model.init(jax.random.PRNGKey(a.seed),jnp.zeros((1,1)),jnp.zeros((1,D)))['params'];state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(3e-4))
key=jax.random.PRNGKey(1000+a.seed);bank=jax.random.normal(jax.random.PRNGKey(70000+a.seed),(N,D))
@jax.jit
def heads(params,z):
 return model.apply({'params':params},jnp.zeros((z.shape[0],1)),z)
@jax.jit
def train(state,key,count):
 def body(_,carry):
  state,key=carry;key,pk=jax.random.split(key)
  if a.mode=='fixed':
   z=jnp.broadcast_to(bank,(B,N,D));mu,ls=heads(state.params,bank);mu=jnp.broadcast_to(mu,z.shape);ls=jnp.broadcast_to(ls,z.shape)
  else:
   z=jax.random.normal(jax.random.fold_in(pk,429),(B,N,D));mu,ls=heads(state.params,z.reshape(-1,D));mu=mu.reshape(z.shape);ls=ls.reshape(z.shape)
  proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),.05)
  act,u,_=proposal.sample(pk,1,'exact');lq=proposal.log_prob(u)-D*jnp.log(SCALE);q=qvalue(SCALE*act)
  u=jax.lax.stop_gradient(u);w=jax.lax.stop_gradient(jax.nn.softmax(q-lq,axis=-1))
  def loss(params):
   mu,ls=heads(params,z.reshape(-1,D));return direct_gmm_nll(mu.reshape(B,N,D),ls.reshape(B,N,D),u,w)[0]
  loss,g=jax.value_and_grad(loss)(state.params)
  return state.apply_gradients(grads=g),key
 return jax.lax.fori_loop(0,count,body,(state,key))
@jax.jit
def samples(params,key):
 zk,ek=jax.random.split(key)
 z=bank[jax.random.randint(zk,(EVAL,),0,N)] if a.mode=='fixed' else jax.random.normal(zk,(EVAL,D))
 mu,ls=heads(params,z)
 return SCALE*jnp.tanh(mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape))
# Independent target-density equation and exact initial prior identity checks.
x=np.linspace(-39,39,22).reshape(11,2).astype(np.float32);expected=logsumexp(-.5*np.square((x[:,None,:]-MEANS)/STD[:,None]).sum(-1)-2*np.log(STD)-np.log(2*np.pi),axis=-1)-np.log(40)
np.testing.assert_allclose(qvalue(jnp.asarray(x)),expected,rtol=2e-5,atol=2e-5)
reference=sample_target(EVAL,451);np.save(out/'reference.npy',reference);np.save(out/'latent_bank.npy',bank)
steps=[0,1000,5000,10000,20000,50000,100000];rows=[];train_seconds=0.;start=time.time()
for i,step in enumerate(steps):
 if i:
  tick=time.time();state,key=train(state,key,step-steps[i-1]);jax.block_until_ready(state.params);train_seconds+=time.time()-tick
 x=np.asarray(samples(state.params,jax.random.PRNGKey(80000+a.seed)));row=metrics(x,reference)
 mu,ls=heads(state.params,bank);row.update(step=step,train_seconds=train_seconds,elapsed_seconds=time.time()-start,sigma_mean=float(jnp.exp(ls).mean()),mu_spread=float(jnp.std(mu)))
 rows.append(row);np.savez_compressed(out/f'step{step}.npz',samples=x,mu=np.asarray(mu),log_sigma=np.asarray(ls),z=np.asarray(bank))
 (out/'metrics.json').write_text(json.dumps(dict(manifest=manifest,rows=rows),indent=2))
 (out/'checkpoint.msgpack').write_bytes(serialization.to_bytes(dict(state=state,key=key,bank=bank,step=step)))
 print(json.dumps(dict(mode=a.mode,seed=a.seed,**{k:v for k,v in row.items() if k not in ['counts','coverage_threshold']})),flush=True)
print('COMPLETE',flush=True)
