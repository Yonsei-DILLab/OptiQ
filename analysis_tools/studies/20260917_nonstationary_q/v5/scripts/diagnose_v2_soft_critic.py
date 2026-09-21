"""Read-only frozen-policy soft-return checks on complete Humanoid episodes.

These on-policy Monte Carlo errors are not uniform critic-error bounds. The
entropy bracket holds in expectation; individual trajectories are not bounds.
Time-limit episodes are reported and excluded instead of silently treating a
timeout as a true terminal. No model or training process is changed.
"""
from pathlib import Path
from functools import partial
import argparse
import json
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
from optiq_dime.soft_improvement import entropy_bracket_sample


def discounted_q_samples(rewards, entropy, temperature, gamma):
    """Q excludes current-action entropy; entropy enters at the NEXT state."""
    values = np.empty(len(rewards))
    values[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        values[i] = rewards[i] + gamma * (values[i + 1] + temperature * entropy[i + 1])
    return values


@partial(jax.jit, static_argnames=("components",))
def action_and_entropy(actor, obs, key, components):
    actions, _, lower, upper = entropy_bracket_sample(actor, obs, key, components)
    return actions, lower, upper


@jax.jit
def q_values(critic, obs, actions):
    inputs = {"params": critic.params, "batch_stats": critic.batch_stats}
    target = {"params": critic.target_params, "batch_stats": critic.target_batch_stats}
    kwargs = {"train": False, "rngs": {"dropout": jax.random.PRNGKey(0)}}
    return (critic.apply_fn(inputs, obs, actions, **kwargs)[..., 0].min(axis=0),
            critic.apply_fn(target, obs, actions, **kwargs)[..., 0].min(axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--step", type=int, default=25000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="soft_critic_diagnosis")
    args = parser.parse_args()
    results = []
    for name in args.run:
        directory = Path(name)
        cfg = OmegaConf.load(directory / "config.json")
        assert cfg.alg.actor.type == "semi_implicit" and cfg.alg.critic.n_atoms == 1
        env = gym.make("Humanoid-v4")
        model = OptiQDIME("MlpPolicy", env, None, 1, cfg)
        checkpoint = next(directory.glob(f"checkpoints/*/actor_state_{args.step}.msgpack"))
        actor = serialization.from_bytes(model.policy.actor_state, checkpoint.read_bytes())
        critic = serialization.from_bytes(model.policy.qf_state,
            (checkpoint.parent / f"critic_state_{args.step}.msgpack").read_bytes())
        episodes = []
        excluded_timeouts = 0
        try:
            for index in range(args.episodes):
                obs, _ = env.reset(seed=940000 + index)
                key = jax.random.PRNGKey(950000 + index)
                observations, actions, rewards, lower, upper = [], [], [], [], []
                done = False
                while not done:
                    key, action_key = jax.random.split(key)
                    action, hl, hu = action_and_entropy(actor,
                        jnp.asarray(obs[None], dtype=jnp.float32), action_key, cfg.alg.actor.entropy_samples)
                    normalized = np.asarray(action)[0]
                    observations.append(obs.copy()); actions.append(normalized)
                    lower.append(float(hl[0])); upper.append(float(hu[0]))
                    obs, reward, terminated, truncated, _ = env.step(model.policy.unscale_action(normalized))
                    rewards.append(float(reward)); done = terminated or truncated
                if truncated and not terminated:
                    excluded_timeouts += 1
                    continue
                rewards, lower, upper = map(np.asarray, (rewards, lower, upper))
                lower_q = discounted_q_samples(rewards, lower, cfg.alg.actor.temperature, cfg.alg.gamma)
                upper_q = discounted_q_samples(rewards, upper, cfg.alg.actor.temperature, cfg.alg.gamma)
                reward_q = discounted_q_samples(rewards, lower, 0., cfg.alg.gamma)
                predicted, target = map(np.asarray, q_values(critic,
                    jnp.asarray(np.stack(observations), dtype=jnp.float32), jnp.asarray(np.stack(actions))))
                episodes.append({"env_seed": 940000 + index, "policy_seed": 950000 + index,
                    "length": len(rewards), "return": float(rewards.sum()),
                    "initial_mc_soft_lower": float(lower_q[0]), "initial_mc_soft_upper": float(upper_q[0]),
                    "initial_mc_reward": float(reward_q[0]), "initial_predicted_q": float(predicted[0]),
                    "initial_target_q": float(target[0]), "mean_entropy_lower": float(lower.mean()),
                    "mean_entropy_upper": float(upper.mean()),
                    "mean_q_minus_mc_lower": float((predicted - lower_q).mean()),
                    "rmse_q_vs_mc_lower": float(np.sqrt(np.mean((predicted - lower_q)**2)))})
            result = {"run": directory.name, "checkpoint_step": args.step,
                "complete_episodes": len(episodes), "excluded_timeouts": excluded_timeouts,
                "episodes": episodes,
                "mean": {k: float(np.mean([r[k] for r in episodes])) for k in episodes[0]
                         if k not in {"env_seed", "policy_seed"}} if episodes else {}}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()
    (ROOT / f"outputs/v2_improvement/{args.output}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
