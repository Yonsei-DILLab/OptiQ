"""Flat-Q check: a Boltzmann teacher should approach uniform action density.

No environment transitions or running experiments are modified. Compare the
original hard point projection with the full-coupling conditional likelihood.
"""
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


def main():
    out=ROOT/'outputs/v2_improvement';out.mkdir(exist_ok=True,parents=True)
    obs=jnp.asarray(np.load(ROOT/'outputs/v2_validation/calibration_observations.npy')[:32])
    records=[]
    for loss in ('pointwise_mse','conditional_ot_nll'):
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs')):
            cfg=compose(config_name='archive/v2/original')
        model=OptiQDIME('MlpPolicy',gym.make('Humanoid-v4'),None,1,cfg)
        actor=model.policy.actor_state
        critic=model.policy.qf_state.replace(apply_fn=flat_q)
        key=jax.random.PRNGKey(203)
        for step in range(1001):
            actor,_,key,metrics=OptiQDIME.update_actor(actor,critic,obs,key,jnp.array([-3600.]),
                16,4,'exact',.8,.5,False,True,1.,False,16.,257,.5,.25,100,'mean','argmax',True,False,loss)
            if step in (0,10,100,500,1000):
                row={'loss':loss,'update':step,**{k:float(metrics[k]) for k in
                    ('policy_entropy_lower','actor_std_mean','actor_std_min','source_ess_absolute',
                     'hard_projection_mass_tv','policy_spread_l2','actor_loss')}}
                assert all(np.isfinite(v) for k,v in row.items() if k!='loss')
                records.append(row);print(json.dumps(row),flush=True)
        model.get_env().close()
    (out/'flat_q_projection.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
