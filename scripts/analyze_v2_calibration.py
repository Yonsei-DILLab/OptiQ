"""Compare completed pilot runs and paired self-inclusive entropy resolutions."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flax import serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from optiq_dime import OptiQDIME
from optiq_dime.semi_implicit import actor_components, conditional_mixture_log_prob
from scripts.calibrate_v2 import entropy_resolution


def main():
    summaries=[]
    for directory in sorted((ROOT/'outputs/v2_calibration').iterdir()):
        if not (directory/'completed.json').exists():
            continue
        cfg = OmegaConf.load(directory/'config.json')
        if cfg.alg.behavior_uniform_probability != 0:
            continue  # Superseded pilots; user requested the original policy only.
        rows=list(csv.DictReader((directory/'logs/progress.csv').open()))
        summary={"directory":str(directory), "temperature":cfg.alg.actor.temperature,
                 "wandb":json.loads((directory/'completed.json').read_text())["wandb_url"]}
        for key in ("source_ess_absolute", "source_ess_min", "policy_entropy_lower", "backup_entropy_lower",
                    "backup_entropy_term", "critic_loss", "actor_std_mean", "reward_mean",
                    "student_action_saturation_fraction", "ot_row_marginal_error"):
            vals=[float(r['train/'+key]) for r in rows if r.get('train/'+key)
                  and float(r.get('time/total_timesteps') or 0)>=15000]
            summary[key+"_late_mean"]=float(np.mean(vals)) if vals else None
        evaluations=list(directory.glob('eval/*/evaluations.npz'))
        ev=np.load(evaluations[0])
        summary['eval_steps']=ev['timesteps'].tolist()
        summary['eval_returns']=ev['results'].mean(-1).tolist()
        summary['eval_lengths']=ev['ep_lengths'].mean(-1).tolist()
        model=OptiQDIME('MlpPolicy',gym.make('Humanoid-v4'),None,1,cfg)
        checkpoint=list(directory.glob('checkpoints/*/actor_state_20000.msgpack'))[0]
        model.policy.actor_state=serialization.from_bytes(model.policy.actor_state,checkpoint.read_bytes())
        probe=list(directory.glob('checkpoints/*/landscape_probe_batch.npz'))[0]
        obs=jnp.asarray(np.load(probe)['observations'])
        paired=[]
        for seed in range(8):
            paired.append(entropy_resolution(model.policy.actor_state,obs,jax.random.PRNGKey(1234+seed)))
        summary['paired_entropy']={k:float(np.mean([r[k] for r in paired])) for k in paired[0]}
        mu,ls=actor_components(model.policy.actor_state,obs,jax.random.PRNGKey(1),16)
        summary['log_std_at_min_fraction']=float(jnp.mean(ls<=cfg.alg.actor.log_std_min+1e-6))
        summary['log_std_at_max_fraction']=float(jnp.mean(ls>=cfg.alg.actor.log_std_max-1e-6))
        summary['actor_std_quantiles']=np.quantile(np.exp(np.asarray(ls)),[0,.01,.1,.5,.9,.99,1]).tolist()
        summary['mu_abs_quantiles']=np.quantile(np.abs(np.asarray(mu)),[.5,.9,.99,1]).tolist()
        model.get_env().close()
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    (ROOT/'outputs/v2_validation/pilot_summary.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
