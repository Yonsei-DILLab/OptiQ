"""Compare two frozen policies under both critics on recent visited states.

This post-hoc diagnostic neither updates parameters nor controls training.
Learned-Q scores on sampled visited states are not true soft-value certificates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True, choices=range(4))
    p.add_argument("--old-step", type=int, required=True)
    p.add_argument("--new-step", type=int, required=True)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 0 < args.old_step < args.new_step:
        p.error("Positive, increasing checkpoint steps required")
    from flax import serialization
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from optiq_dime import OptiQDIME
    from optiq_dime.soft_improvement import candidate_gap_samples

    manifest = json.loads((ROOT / "outputs/v2_improvement/confirmation_manifest.json").read_text())
    items = [i for i in manifest["runs"] if i["method"] == "v2" and i["seed"] == args.seed]
    assert len(items) == 1
    directory = Path(items[0]["directory"])
    cfg = OmegaConf.load(directory / "config.json")
    assert cfg.seed == args.seed and cfg.runtime.git_commit == items[0]["commit"]
    assert cfg.alg.actor.type == "semi_implicit" and cfg.alg.critic.n_atoms == 1
    assert cfg.alg.behavior_uniform_probability == 0 and cfg.env_name == "Humanoid-v4"
    environment = gym.make(cfg.env_name)
    model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
    steps = [args.old_step, args.new_step]
    actors, critics, provenance = {}, {}, {}
    for step in steps:
        actor_path = next(directory.glob(f"checkpoints/*/actor_state_{step}.msgpack"))
        critic_path = actor_path.parent / f"critic_state_{step}.msgpack"
        actor_data, critic_data = actor_path.read_bytes(), critic_path.read_bytes()
        actors[step] = serialization.from_bytes(model.policy.actor_state, actor_data)
        critics[step] = serialization.from_bytes(model.policy.qf_state, critic_data)
        provenance[str(step)] = {
            "actor": str(actor_path), "actor_sha256": hashlib.sha256(actor_data).hexdigest(),
            "critic": str(critic_path), "critic_sha256": hashlib.sha256(critic_data).hexdigest()}
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "training_seed": args.seed,
              "old_step": args.old_step, "new_step": args.new_step, "seed_base": args.seed_base,
              "source_commit": items[0]["commit"],
              "diagnostic_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__,
              "checkpoints": provenance, "state_banks": [], "scores": [], "complete": False,
              "limitations": ["Four trajectories per policy; sampled states do not cover the global state space.",
                  "States are sampled equally per trajectory, including states near early termination.",
                  "MC standard errors condition on the fixed models and selected states.",
                  "The gap is new-policy lower minus old-policy upper under the same learned min-Q critic.",
                  "A long checkpoint-to-checkpoint change is not an individual accepted training update.",
                  "Policy/critic rankings do not identify a causal mechanism or establish true soft improvement."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    banks, saved = {}, {}
    try:
        for step in steps:
            model.policy.actor_state = actors[step]
            selected_states, sources, episodes = [], [], []
            for episode in range(4):
                obs, _ = environment.reset(seed=args.seed_base + episode)
                model.policy.key = jax.random.PRNGKey(args.seed_base + 10000 + episode)
                trajectory, total, done = [], 0., False
                while not done:
                    trajectory.append(obs.copy())
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = environment.step(action)
                    total += float(reward)
                    done = terminated or truncated
                indices = np.linspace(0, len(trajectory)-1, min(32, len(trajectory)), dtype=int)
                selected_states.extend(np.stack(trajectory)[indices])
                sources.extend((episode, int(index)) for index in indices)
                episodes.append({"episode": episode, "env_seed": args.seed_base+episode,
                                 "policy_seed": args.seed_base+10000+episode, "return": total,
                                 "length": len(trajectory), "terminated": bool(terminated),
                                 "truncated": bool(truncated)})
            states = np.asarray(selected_states, dtype=np.float32)
            banks[step] = states
            saved[f"observations_{step}"] = states
            saved[f"episode_step_{step}"] = np.asarray(sources)
            record["state_banks"].append({"policy_step": step, "state_count": len(states),
                                          "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                                          "episodes": episodes})
        state_file = args.output.with_name(args.output.stem + "_states.npz")
        np.savez_compressed(state_file, **saved)
        record["state_file"] = str(state_file)
        temperature = float(cfg.alg.actor.temperature)
        components = int(cfg.alg.actor.entropy_samples)
        atoms = jnp.asarray([cfg.alg.critic.v_min])

        @jax.jit
        def sample_gaps(old, new, critic, observations, key):
            return candidate_gap_samples(old, new, critic, observations, key,
                                         temperature, atoms, components, 16)

        for bank_step, states in banks.items():
            observations = jnp.asarray(states)
            for critic_step in steps:
                chunks = []
                for chunk in range(16):
                    key = jax.random.fold_in(jax.random.PRNGKey(args.seed_base+20000), chunk)
                    chunks.append(np.asarray(sample_gaps(actors[steps[0]], actors[steps[1]],
                                                         critics[critic_step], observations, key)))
                samples = np.concatenate(chunks)
                assert samples.shape == (256, len(states)) and np.isfinite(samples).all()
                state_means, draw_means = samples.mean(axis=0), samples.mean(axis=1)
                result = {"state_bank_policy_step": bank_step, "critic_step": critic_step,
                          "draws_per_state": 256, "temperature": temperature, "components": components,
                          "mean_new_minus_old_soft_score": float(samples.mean()),
                          "conditional_mc_standard_error": float(draw_means.std(ddof=1)/np.sqrt(256)),
                          "negative_state_mean_fraction": float((state_means < 0).mean()),
                          "state_mean_quantiles": np.quantile(state_means, [0, .1, .5, .9, 1]).tolist(),
                          "state_mean_gaps": state_means.tolist()}
                record["scores"].append(result)
                record["updated_utc"] = datetime.now(timezone.utc).isoformat()
                record["complete"] = len(record["scores"]) == 4
                args.output.write_text(json.dumps(record, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "state_mean_gaps"}), flush=True)
    finally:
        model.get_env().close()


if __name__ == "__main__":
    main()
