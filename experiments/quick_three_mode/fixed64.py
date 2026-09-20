"""Fixed training latent bank of 64; 64 IID proposal samples each update."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from experiments.gmm_gradient_interference.core import initialize,base,draw,heads,basin_prob,problems,OBS,QARG
from optiq_dime.semi_implicit import ConditionalGaussianProposal
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)/f'fixed64_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
state,key=initialize(a.seed);z=jax.random.normal(jax.random.PRNGKey(70000+a.seed),(64,1));engine=base(64,64)
@jax.jit
def train(state,key,count):
 def body(_,carry):
  state,key=carry;key,pk=jax.random.split(key)
  mu,ls=heads(state.params,z)
  proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu[None]),jax.lax.stop_gradient(ls[None]),.05)
  b,u,indices=proposal.sample(pk,1,'exact');lq=proposal.log_prob(u)
  q=problems.analytic_q_jax(b,QARG);w=jax.nn.softmax(q/.25-lq,axis=-1)
  teacher=jax.tree_util.tree_map(jax.lax.stop_gradient,dict(obs=OBS,z=z[None],u=u,w=w))
  state,_,_=engine['update'](state,teacher)
  return state,key
 return jax.lax.fori_loop(0,count,body,(state,key))
@jax.jit
def fixed_draw(params,key):
 ck,ek=jax.random.split(key);mu,ls=heads(params,z);idx=jax.random.randint(ck,(32768,),0,64)
 return jnp.tanh(mu[idx,0]+jnp.exp(ls[idx,0])*jax.random.normal(ek,(32768,)))
rows=[];t0=time.time();edges=np.linspace(-1,1,257);target=np.diff(problems.analytic_cdf(edges,problems.schedule('prefix',1)))
steps=[0,100,500,1000,2000,5000]
for i,step in enumerate(steps):
 if i:state,key=train(state,key,step-steps[i-1]);jax.block_until_ready(state.params)
 samples=np.asarray(fixed_draw(state.params,jax.random.PRNGKey(80000+a.seed)))
 fresh=np.asarray(draw(state.params,jax.random.PRNGKey(80000+a.seed),32768))
 mu,ls=heads(state.params,z);probs=np.asarray(basin_prob(mu,ls));hist=np.histogram(samples,edges)[0]/len(samples)
 mass=np.histogram(samples,[-1,-.3,.3,1])[0]/len(samples)
 row=dict(step=step,mass=mass.tolist(),tv=float(abs(hist-target).sum()/2),specialist_fraction=float((probs.max(1)>=.8).mean()),fresh_z_tv=float(abs(np.histogram(fresh,edges)[0]/len(fresh)-target).sum()/2),seconds=time.time()-t0)
 rows.append(row)
 (out/'metrics.json').write_text(json.dumps(dict(source_commit=sha,n=64,m=64,seed=a.seed,fixed_training_latents=True,evaluation='uniform mixture over fixed 64 z; fresh-z evaluation saved separately',rows=rows),indent=2))
 np.savez_compressed(out/f'step{step}.npz',samples=samples,fresh_samples=fresh,mu=np.asarray(mu),ls=np.asarray(ls),probs=probs,z=np.asarray(z))
 print(json.dumps(dict(seed=a.seed,**row)),flush=True)
print('COMPLETE',flush=True)
