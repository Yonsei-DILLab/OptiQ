"""Read-only finite-M entropy diagnostic on frozen-policy visited states.

The entropy inequalities concern expectations. Monte Carlo gaps on these
states are not uniform entropy or critic-error certificates.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
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


@partial(jax.jit, static_argnames=("components",))
def entropy_draws(actor, observations, keys, components):
    _, _, lower, upper = jax.vmap(
        lambda key: entropy_bracket_sample(actor, observations, key, components)
    )(keys)
    return lower, upper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_manifest.json")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--draws", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=1050000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.step, args.episodes, args.states) < 1 or args.draws < 16 or args.draws % 16:
        p.error("Positive step/episodes/states; draws must be a positive multiple of 16")
    manifest = json.loads(args.manifest.read_text())
    items = sorted([i for i in manifest["runs"] if i["method"] == "v2"], key=lambda i: i["seed"])
    assert [i["seed"] for i in items] == [0, 1, 2, 3]
    output = args.output or ROOT / f"outputs/v2_improvement/entropy_diagnostic_{args.step:07d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "step": args.step,
              "diagnostic_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "devices": [str(d) for d in jax.devices()], "seed_base": args.seed_base,
              "draws_per_state": args.draws, "results": [], "complete": False,
              "limitations": ["Visited states from four trajectories per checkpoint are not global state coverage.",
                  "MC standard error is conditional on the selected states and fixed model.",
                  "A small sample gap does not certify a uniform entropy bound or policy improvement."]}
    for item in items:
        directory = Path(item["directory"])
        cfg = OmegaConf.load(directory / "config.json")
        assert cfg.seed == item["seed"] and cfg.runtime.git_commit == item["commit"]
        assert cfg.env_name == "Humanoid-v4" and cfg.alg.actor.type == "semi_implicit"
        checkpoint = next(directory.glob(f"checkpoints/*/actor_state_{args.step}.msgpack"))
        payload = checkpoint.read_bytes()
        environment = gym.make(cfg.env_name)
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        try:
            model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, payload)
            observations, episodes = [], []
            for index in range(args.episodes):
                env_seed, policy_seed = args.seed_base+index, args.seed_base+10000+index
                obs, _ = environment.reset(seed=env_seed)
                model.policy.key = jax.random.PRNGKey(policy_seed)
                length, total, done = 0, 0., False
                while not done:
                    observations.append(obs.copy())
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = environment.step(action)
                    total += float(reward)
                    length += 1
                    done = terminated or truncated
                episodes.append({"env_seed": env_seed, "policy_seed": policy_seed, "length": length,
                                 "return": total, "terminated": bool(terminated), "truncated": bool(truncated)})
            selected = np.linspace(0, len(observations)-1, min(args.states, len(observations)), dtype=int)
            states = np.stack(observations)[selected].astype(np.float32)
            keys = jax.random.split(jax.random.PRNGKey(args.seed_base+20000), args.draws)
            components = int(cfg.alg.actor.entropy_samples)
            lower, upper = [], []
            for start in range(0, args.draws, 16):
                lo, hi = entropy_draws(model.policy.actor_state, jnp.asarray(states), keys[start:start+16], components)
                lower.append(np.asarray(lo)); upper.append(np.asarray(hi))
            lower, upper = np.concatenate(lower), np.concatenate(upper)
            assert lower.shape == upper.shape == (args.draws, len(states))
            assert np.isfinite(lower).all() and np.isfinite(upper).all()
            gap = upper-lower
            # Independent action/component draws; condition on the chosen states.
            draw_mean_gaps = gap.mean(axis=1)
            result = {"seed": item["seed"], "run": directory.name, "source_commit": item["commit"],
                      "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                      "components": components, "temperature": float(cfg.alg.actor.temperature),
                      "selected_state_indices": selected.tolist(),
                      "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(), "episodes": episodes,
                      "mean_entropy_lower": float(lower.mean()), "mean_entropy_upper": float(upper.mean()),
                      "mean_gap": float(gap.mean()),
                      "conditional_mc_standard_error_gap": float(draw_mean_gaps.std(ddof=1)/np.sqrt(args.draws)),
                      "temperature_times_mean_gap": float(cfg.alg.actor.temperature*gap.mean()),
                      "state_mean_gaps": gap.mean(axis=0).tolist(),
                      "state_mean_entropy_lower": lower.mean(axis=0).tolist()}
            record["results"].append(result)
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            record["complete"] = len(record["results"]) == 4
            output.write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps({k: v for k, v in result.items() if k not in {
                "episodes", "selected_state_indices", "state_mean_gaps", "state_mean_entropy_lower"}}), flush=True)
        finally:
            model.get_env().close()


if __name__ == "__main__":
    main()
