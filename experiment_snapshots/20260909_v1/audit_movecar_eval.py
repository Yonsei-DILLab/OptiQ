import json,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from analysis_boltzmann.movecar import make_agent,dynamics,reward
from optiq_dime.policy import OptiQPolicy
root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(exist_ok=False)
records=[]
for method,seeds in [('optiq',[0,1,3]),('ddpg',[0]),('sd2',[0]),('td3',[0]),('sd3',[0])]:
 for seed in seeds:
  agent=make_agent(method,seed); agent.load(root/'runs'/f'movecar_{method}_seed{seed}'/'checkpoint_1000000.msgpack')
  modes=[('original10',10,False),('stochastic1000',1000,False),('zero_latent',10,True)] if method=='optiq' else [('original10',10,False)]
  for label,n,zero in modes:
   key=jax.random.PRNGKey(seed+90000); x=np.full((n,1),8.,np.float32); xs=[x[:,0].copy()]; actions=[]; rs=[]
   for t in range(100):
    key,ak=jax.random.split(key)
    a=np.asarray(OptiQPolicy.sample_action(agent.model.policy.actor_state,jnp.asarray(x),ak,True)) if zero else np.asarray(agent.act(x,ak))
    x=dynamics(x,a).astype('float32'); xs.append(x[:,0].copy()); actions.append(a[:,0]); rs.append(reward(x)[:,0])
   xs=np.array(xs).T; actions=np.array(actions).T; rs=np.array(rs).T; returns=rs.sum(1)
   high=rs==2; hit=high.any(1); first=np.where(hit,high.argmax(1)+1,101); low=(rs==1).sum(1)
   # Exact loss decomposition: return=2*(101-first)-2*missed_after+low.
   missed=np.array([np.sum(~high[i,first[i]-1:]) if hit[i] else 0 for i in range(n)])
   arrival_loss=2*(first-7); residence_loss=2*missed
   assert np.all(188-returns==arrival_loss+residence_loss-low)
   np.savez_compressed(out/f'{method}_{seed}_{label}.npz',positions=xs,actions=actions,rewards=rs)
   result={'method':method,'seed':seed,'mode':label,'episodes':n,'return_mean':float(returns.mean()),'return_std':float(returns.std(ddof=1)),'return_min':float(returns.min()),'return_max':float(returns.max()),'optimal_fraction':float(np.mean(returns==188)),'first_high_mean':float(first.mean()),'arrival_loss':float(arrival_loss.mean()),'residence_loss':float(residence_loss.mean()),'low_reward_credit':float(low.mean()),'original_returns':returns.tolist() if n==10 else None}
   records.append(result); print(json.dumps(result),flush=True)
(out/'summary.json').write_text(json.dumps(records,indent=2))
