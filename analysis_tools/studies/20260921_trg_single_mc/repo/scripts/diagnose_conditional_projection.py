"""Can the projection expand a flat target and contract a narrow known target?"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gymnasium as gym
from hydra import compose,initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from optiq_dime import OptiQDIME


def flat_q(variables,obs,actions,**kwargs):
    return jnp.zeros((2,len(obs),1),dtype=obs.dtype)


def quadratic_q(variables,obs,actions,**kwargs):
    # At T=.5 the exact action target is a box-truncated N(0,.15^2 I).
    values=-.5*.5*jnp.sum((actions/.15)**2,axis=-1)
    return jnp.broadcast_to(values[None,:,None],(2,len(obs),1))


def main():
    out=ROOT/'outputs/v2_improvement'
    obs=jnp.asarray(np.load(ROOT/'outputs/v2_validation/calibration_observations.npy')[:32])
    result=[]
    for target,q_fn in [('flat',flat_q),('narrow_gaussian',quadratic_q)]:
        for proposal,floor in [('realized_kde',.8),('conditional_mixture',.8),('conditional_mixture',.05)]:
            with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs')):
                cfg=compose(config_name='mujoco_v2')
            model=OptiQDIME('MlpPolicy',gym.make('Humanoid-v4'),None,1,cfg)
            actor=model.policy.actor_state;critic=model.policy.qf_state.replace(apply_fn=q_fn)
            key=jax.random.PRNGKey(203)
            for step in range(1001):
                actor,_,key,m=OptiQDIME.update_actor(actor,critic,obs,key,jnp.array([-3600.]),
                    16,4,'exact',floor,.5,False,True,1.,False,16.,257,.5,.25,100,'mean','argmax',
                    True,False,'conditional_ot_nll',proposal)
                if step in (0,100,1000):
                    row={'target':target,'proposal':proposal,'floor':floor,'update':step,
                        **{k:float(m[k]) for k in ('policy_entropy_lower','actor_std_mean','source_ess_absolute','policy_spread_l2')}}
                    result.append(row);print(json.dumps(row),flush=True)
            model.get_env().close()
    (out/'conditional_projection_diagnosis.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
