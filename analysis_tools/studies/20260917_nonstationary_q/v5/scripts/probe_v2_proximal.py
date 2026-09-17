"""Read-only candidate ESS and held-out sampled soft gain on finite checkpoints.

No environment training, replay writes or checkpoint mutation. Fresh states
come from the saved policy; each candidate is a single supervised OT update.
Scores use a learned frozen critic and are not true policy-improvement bounds.
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flax import serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from optiq_dime import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from optiq_dime.soft_improvement import candidate_gap_samples


def main():
    base=ROOT/'outputs/v2_improvement'
    manifest=json.loads((base/'finite_screen_manifest.json').read_text())
    records=[]
    for run in manifest['runs']:
        directory=Path(run['directory']); cfg=OmegaConf.load(directory/'config.json')
        env=gym.make('Humanoid-v4'); model=OptiQDIME('MlpPolicy',env,None,1,cfg)
        ap=next(directory.glob('checkpoints/*/actor_state_250000.msgpack'))
        qp=ap.parent/'critic_state_250000.msgpack'
        actor=serialization.from_bytes(model.policy.actor_state,ap.read_bytes())
        critic=serialization.from_bytes(model.policy.qf_state,qp.read_bytes())
        observations=[]; resets=0
        obs,_=env.reset(seed=1210000)
        key=jax.random.PRNGKey(1220000)
        for step in range(1024):
            if step%16==0: observations.append(obs.copy())
            key,ak=jax.random.split(key)
            action=OptiQPolicy.sample_action(actor,jnp.asarray(obs[None],dtype=jnp.float32),ak)
            obs,_,terminated,truncated,_=env.step(model.policy.unscale_action(np.asarray(action)[0]))
            if terminated or truncated:
                resets+=1; obs,_=env.reset(seed=1210000+resets)
        bank=np.stack(observations).astype(np.float32)
        np.save(base/f'proximal_probe_states_s{run["seed"]}.npy',bank)
        # Disjoint update and validation state indices; new paired action noise.
        train_obs=jnp.asarray(bank[::2]); heldout_obs=jnp.asarray(bank[1::2])
        for fraction in (0.,.25,.5):
            draws=[]
            for replicate in range(8):
                candidate,loss,_,metrics=OptiQDIME.update_actor(actor,critic,train_obs,
                    jax.random.PRNGKey(1230000+replicate),jnp.zeros(1),16,4,'exact',.05,.5,
                    False,True,1.,False,16.,257,.1,.25,100,'min','argmax',True,False,
                    'conditional_ot_nll','conditional_mixture',fraction)
                gaps=np.asarray(candidate_gap_samples(actor,candidate,critic,heldout_obs,
                    jax.random.PRNGKey(1240000+replicate),.1,jnp.zeros(1),16,64))
                draws.append({'replicate':replicate,'ess':float(metrics['source_ess_absolute']),
                    'eta':float(metrics.get('proximal_fraction_mean',1.)),
                    'loss':float(loss),'heldout_sampled_gain':float(gaps.mean()),
                    'heldout_draw_se':float(gaps.mean(axis=1).std(ddof=1)/np.sqrt(len(gaps)))})
            row={'seed':run['seed'],'checkpoint_step':250000,'ess_fraction':fraction,
                 'actor_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),
                 'critic_sha256':hashlib.sha256(qp.read_bytes()).hexdigest(),
                 'state_bank_sha256':hashlib.sha256(bank.tobytes()).hexdigest(),'replicates':draws,
                 'means':{k:float(np.mean([r[k] for r in draws])) for k in draws[0] if k!='replicate'}}
            records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='replicates'}),flush=True)
        model.get_env().close()
        output={'generated_utc':datetime.now(timezone.utc).isoformat(),'records':records,
                'scope':'Frozen capped finite policies; one OT update with retained Adam state. Learned-Q held-out sample gains, not environment returns or an improvement certificate.',
                'state_collection':{'steps':1024,'sample_every':16,'env_seed_start':1210000,'policy_seed':1220000}}
        (base/'proximal_frozen_probe.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
