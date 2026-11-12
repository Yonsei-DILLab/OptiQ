"""Read-only target-critic evaluation of the actual TD action sampler on fixed states."""
from pathlib import Path
import json,hashlib
import numpy as np
import jax,jax.numpy as jnp
import flax.serialization
from omegaconf import OmegaConf
from gymnasium import spaces
from ibolt.policy import IBOLTPolicy
P=Path(__file__).resolve().parent
assert jax.default_backend()=='cpu'
spec=json.loads((P/'probe-spec.json').read_text());raw=np.load(P/'raw-state-bank.npy')
cfg=OmegaConf.create(json.loads((P/spec['runs'][0]['id']/'mujoco-config.json').read_text()))
policy=IBOLTPolicy(spaces.Box(-np.inf,np.inf,(28,),dtype=np.float32),spaces.Box(-1.,1.,(4,),dtype=np.float32),cfg=cfg)
policy.build(jax.random.PRNGKey(15171),lambda _:3e-4,3e-4)
@jax.jit
def probe(actor,critic,obs,key):
    # Same action sampler and target critic as update_critic, no candidate selection.
    a=IBOLTPolicy.sample_action(actor,obs,key)
    q=critic.apply_fn({'params':critic.target_params,'batch_stats':critic.target_batch_stats},obs,a,train=False)[...,0]
    return q
rows=[]
for step in [100096,500224,1000192,1500160]:
 for run in spec['runs']:
    p=P/run['id']/'checkpoints'
    a=flax.serialization.from_bytes(policy.actor_state,(p/f'actor_state_{step}.msgpack').read_bytes())
    c=flax.serialization.from_bytes(policy.qf_state,(p/f'critic_state_{step}.msgpack').read_bytes())
    n=json.loads((p/f'normalizer_{step:09d}.json').read_text())
    obs=np.clip((np.clip(raw,-200,200)-np.asarray(n['mean']))/np.maximum(np.sqrt(np.asarray(n['variance'])),.01),-5,5).astype(np.float32)
    q=np.concatenate([np.asarray(probe(a,c,jnp.asarray(obs),jax.random.PRNGKey(2026111200+i))) for i in range(16)],axis=1)
    assert np.isfinite(q).all()
    mean=q.mean(0);low=q.min(0);gap=np.abs(q[0]-q[1])
    rows.append(dict(id=run['id'],seed=run['seed'],trained_M=run['M'],step=step,target_mean_q=float(mean.mean()),target_min_q=float(low.mean()),target_twin_gap=float(gap.mean()),gamma_times_mean_minus_min=float(.99*(mean-low).mean()),fraction_target_min_below_minus100=float((low < -100).mean())))
out=dict(read_only=True,optimizer_updates=0,backend='cpu',states=128,action_draws_per_state=16,state_bank_sha256=hashlib.sha256((P/'raw-state-bank.npy').read_bytes()).hexdigest(),method='Actual TD policy action sampler and target critic at fixed symmetric state bank. No historical replay batch; no intervened training. Mean-vs-min difference is an algebraic diagnostic, not proof of its causal contribution to learned Q error.',rows=rows)
(P/'td-probe-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
