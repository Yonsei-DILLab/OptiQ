"""Read-only candidate acceptance probes; never changes a running experiment."""
from pathlib import Path
import json
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flax import serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from optiq_dime import OptiQDIME
from optiq_dime.soft_improvement import candidate_gap_samples, sampled_soft_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=32)
    parser.add_argument("--output", default="soft_guard_probe")
    parser.add_argument("--checkpoint", type=int, default=50000)
    args = parser.parse_args()
    results = []
    for directory in sorted((ROOT / "outputs/v2_conditional_screen").glob("*s0_*")):
        cfg = OmegaConf.load(directory / "config.json")
        model = OptiQDIME("MlpPolicy", gym.make("Humanoid-v4"), None, 1, cfg)
        checkpoint = next(directory.glob(f"checkpoints/*/actor_state_{args.checkpoint}.msgpack"))
        step = int(checkpoint.stem.split("_")[-1])
        old = serialization.from_bytes(model.policy.actor_state, checkpoint.read_bytes())
        critic = serialization.from_bytes(model.policy.qf_state,
            (checkpoint.parent / f"critic_state_{step}.msgpack").read_bytes())
        observations = jnp.asarray(np.load(checkpoint.parent / "landscape_probe_batch.npz")["observations"])
        training, validation = observations[:32], observations[32:32 + args.validation_batch]
        assert len(validation) == args.validation_batch
        ac = cfg.alg.actor
        atoms = jnp.array([cfg.alg.critic.v_min])

        @jax.jit
        def probe(key):
            proposal_key, check_key, confirmation_key = jax.random.split(key, 3)
            new, _, _, _ = OptiQDIME.update_actor(old, critic, training, proposal_key, atoms,
                ac.num_policy_samples, ac.proposals_per_policy_sample, ac.proposal_sampling_mode,
                ac.proposal_std, ac.proposal_clip, ac.include_anchor, ac.density_correction,
                ac.density_beta, ac.adaptive_density_beta, ac.minimum_source_ess,
                ac.density_beta_grid_size, ac.temperature, ac.sinkhorn_epsilon,
                ac.sinkhorn_iterations, "min", ac.transport_target_mode,
                True, ac.normalize_ot_cost, ac.distillation_loss, ac.teacher_distribution)
            _, metrics = sampled_soft_update(old, new, critic, validation, check_key,
                ac.temperature, atoms, 16, args.draws, 2.)
            confirmation = candidate_gap_samples(old, new, critic, validation,
                confirmation_key, ac.temperature, atoms, 16, 64)
            baseline = candidate_gap_samples(old, old, critic, validation,
                check_key, ac.temperature, atoms, 16, args.draws)
            return {**metrics, "independent_64_draw_gap": confirmation.mean(),
                    "identical_actor_bracket_gap": baseline.mean()}

        rows = []
        for trial in range(16):
            row = {k: float(v) for k, v in probe(jax.random.PRNGKey(91200 + trial)).items()}
            rows.append(row)
        accepted = [r for r in rows if r["soft_guard_accepted"]]
        result = {"run": directory.name, "checkpoint_step": step, "trials": rows,
            "validation_states": len(validation), "draws": args.draws,
            "acceptance_fraction": len(accepted) / len(rows),
            "confirmation_positive_fraction_among_accepted": (
                np.mean([r["independent_64_draw_gap"] > 0 for r in accepted]).item()
                if accepted else None)}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)
        model.get_env().close()
    (ROOT / f"outputs/v2_improvement/{args.output}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
