"""GPU-only initial proposal/OT calibration; no training run is modified."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np

from optiq_dime import OptiQDIME
from optiq_dime.semi_implicit import actor_components, conditional_mixture_log_prob
from run_optiq_dime import validate_config


def entropy_resolution(actor, obs, key):
    zk, ek = jax.random.split(key)
    mu, ls = actor_components(actor, obs, zk, 64)
    u = mu[:,0] + jnp.exp(ls[:,0])*jax.random.normal(ek, mu[:,0].shape)
    result = {}
    for m in (1, 4, 16, 64):
        entropy = -conditional_mixture_log_prob(u[:,None], mu[:,:m], ls[:,:m])[:,0]
        result[f"entropy_M{m}"] = float(entropy.mean())
    return result


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError("Calibration requires GPU")
    out = ROOT / "outputs/v2_validation"
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make("Humanoid-v4")
    env.action_space.seed(90210)
    obs, _ = env.reset(seed=90210)
    observations, rewards = [], []
    for t in range(1024):
        if t % 8 == 0:
            observations.append(obs.copy())
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    obs = jnp.asarray(np.asarray(observations, dtype=np.float32))
    np.save(out / "calibration_observations.npy", np.asarray(obs))
    records = []
    for std in (.3, .5, 1.):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="mujoco_v2", overrides=[f"alg.actor.initial_log_std={np.log(std)}"])
        validate_config(cfg)
        model = OptiQDIME("MlpPolicy", gym.make("Humanoid-v4"), None, 1, cfg)
        a = cfg.alg.actor
        entropy = entropy_resolution(model.policy.actor_state, obs, jax.random.PRNGKey(920))
        for h in (.2, .4, .6, .8, 1.):
            for epsilon in (.05, .1, .25):
                _, _, _, metrics = OptiQDIME.update_actor(
                    model.policy.actor_state, model.policy.qf_state, obs, jax.random.PRNGKey(90210),
                    jnp.array([-3600.]), 16, 4, "exact", h, .5, False, True, 1., False, 16., 257,
                    .25, epsilon, 100, "mean", "argmax", True, False)
                row = {"initial_std":std, "h":h, "epsilon":epsilon, **entropy,
                       **{k:float(v) for k,v in metrics.items()}}
                if not all(np.isfinite(v) for v in row.values()):
                    raise FloatingPointError("Nonfinite calibration")
                records.append(row)
                print(json.dumps({k:row[k] for k in ("initial_std","h","epsilon","source_ess_absolute",
                    "policy_entropy_lower","teacher_action_saturation_fraction","ot_row_marginal_error")}), flush=True)
        model.get_env().close()
    (out/"initial_calibration.json").write_text(json.dumps({
        "state_count":len(obs), "random_behavior_reward_mean":float(np.mean(rewards)),
        "records":records, "note":"Untrained networks: bandwidth/entropy/numerics only; not evidence of final return."
    }, indent=2))


if __name__ == "__main__":
    main()
