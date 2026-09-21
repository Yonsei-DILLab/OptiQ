"""Targeted checks against the actual successful GMM40 functions and TD formula."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from circuit import log_prob_from_output, q_from_output, sample_from_output
from algorithms.spline_energy.raw_energy import (
    ConditionalRawEnergyCircuit, fixed_q_forward_energy_loss, output_from_factors)
from train import make_update

ROOT = Path(__file__).resolve().parents[3]


def original_gmm():
    path = ROOT / "benchmarks/gmm40/spline_energy"
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("original_gmm40_spline", path / "spline_energy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_parity():
    original = original_gmm()
    state, _ = original.initialize_circuit(2)  # Successful rank64 / knots129.
    model = ConditionalRawEnergyCircuit(2, initialization_seed=2, hidden_dims=(16, 16))
    observations = jnp.zeros((96, 3))
    params = model.init(jax.random.PRNGKey(3), observations)["params"]
    assert "value" not in params
    output = model.apply({"params": params}, observations)
    actions = jax.random.uniform(jax.random.PRNGKey(17), (96, 2), minval=-.999, maxval=.999)
    expected_q = original.circuit_q(state.params, actions)
    expected_z = original.circuit_partition(state.params)[0]
    np.testing.assert_allclose(output["raw_log_roots"][0], state.params["height"], atol=2e-6)
    np.testing.assert_allclose(output["raw_log_leaves"][0], state.params["leaves"], atol=2e-6)
    np.testing.assert_allclose(output["log_partition"], expected_z, atol=2e-6)
    np.testing.assert_allclose(q_from_output(output, actions, .25), .25 * expected_q, atol=3e-6)
    np.testing.assert_allclose(log_prob_from_output(output, actions),
                               original.circuit_logp(state.params, actions), atol=4e-6)

    target = -.5 * jnp.square(actions - .25).sum(-1)
    logb = jnp.full((len(actions),), -np.log(4.))
    weights = jax.lax.stop_gradient(jnp.exp(target - logb))
    def old_loss(p):
        return jnp.exp(original.circuit_partition(p)[0]) - jnp.mean(weights * original.circuit_q(p, actions))
    def new_loss(p):
        out = output_from_factors(jnp.broadcast_to(p["height"], (len(actions), 64)),
                                 jnp.broadcast_to(p["leaves"], (len(actions), 64, 2, 129)), .25)
        return fixed_q_forward_energy_loss(out, actions, target, logb)
    old_value, old_grad = jax.value_and_grad(old_loss)(state.params)
    new_value, new_grad = jax.value_and_grad(new_loss)(state.params)
    np.testing.assert_allclose(old_value, new_value, rtol=2e-6, atol=2e-6)
    gradient_error = 0.
    for a, b in zip(jax.tree_util.tree_leaves(old_grad), jax.tree_util.tree_leaves(new_grad)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
        gradient_error = max(gradient_error, float(jnp.max(jnp.abs(a - b))))

    # Same random draws must produce the reference sampler's actions.
    key = jax.random.PRNGKey(21)
    expected_samples = original.circuit_sample(state.params, key, len(actions), parallel_root=True)
    actual_samples = sample_from_output(output, key)
    np.testing.assert_allclose(actual_samples, expected_samples, atol=2e-6)
    # With b=pi, copying Direct's teacher from our OWN tied Q is self-imitation.
    logits = q_from_output(output, actions, .25) / .25 - log_prob_from_output(output, actions)
    np.testing.assert_allclose(logits, output["log_partition"], atol=4e-6)
    return {"rank": 64, "knots": 129, "oracle_loss": float(new_value),
            "max_factor_gradient_error": gradient_error,
            "max_sampler_action_error": float(jnp.max(jnp.abs(actual_samples - expected_samples))),
            "self_teacher_logit_range": float(jnp.ptp(logits))}


def td_path():
    model = ConditionalRawEnergyCircuit(1, rank=4, knots=17, hidden_dims=(16, 16))
    # A nonzero context exercises the state encoder. With both zero context
    # and zero head kernels, only the state-free output biases can move.
    observations = jnp.broadcast_to(jnp.asarray([.2, -.3]), (128, 2))
    params = model.init(jax.random.PRNGKey(5), observations)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))
    actions = jnp.linspace(-.99, .99, 128)[:, None]
    batch = {"obs": observations, "next_obs": observations + .2, "actions": actions,
             "rewards": -jnp.square(actions[:, 0] - .4),
             "not_terminal": jnp.arange(128) % 2}
    key = jax.random.PRNGKey(31)
    current = model.apply({"params": params}, batch["next_obs"])
    sampled = sample_from_output(current, key)
    manual_target = batch["rewards"] + .99 * batch["not_terminal"] * q_from_output(current, sampled, .25)
    original_q = q_from_output(model.apply({"params": params}, observations), actions, .25)
    update = make_update(model, .25, .99, .005, 10., loss_kind="mse", backup_mode="td")
    new_state, new_target, metrics = update(state, params, batch, key)
    np.testing.assert_allclose(metrics["train/target_mean"], manual_target.mean(), rtol=2e-6)
    np.testing.assert_allclose(metrics["train/loss"], jnp.square(original_q - manual_target).mean(), rtol=2e-6)
    assert "train/v_mean" not in metrics
    assert "train/alpha_log_partition_mean" in metrics
    target_gradient = jax.grad(lambda p: update(state, p, batch, key)[2]["train/loss"])(params)
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(target_gradient)) == 0.
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((new_state.params, new_target, metrics)))

    # Deterministic terminal contextual problem checks the actual optimization
    # path, independent of a moving bootstrap or claimed MuJoCo performance.
    batch["not_terminal"] = jnp.zeros((128,))
    target_params = params
    first = float(update(state, target_params, batch, key)[2]["train/loss"])
    for step in range(192):
        state, target_params, info = update(state, target_params, batch, jax.random.fold_in(key, step))
    last = float(info["train/loss"])
    assert np.isfinite(last) and last < first * .5, (first, last)
    return {"plain_td_target_mean": float(manual_target.mean()),
            "target_gradient_max": 0., "contextual_initial_mse": first,
            "contextual_final_mse": last, "contextual_updates": 192}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"passed": True, "gmm40_parity": reference_parity(), "td_path": td_path(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": sys.argv,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__), ROOT / "algorithms/spline_energy/raw_energy.py",
                            ROOT / "algorithms/spline_energy/model.py", Path(__file__).with_name("train.py"),
                            ROOT / "benchmarks/gmm40/spline_energy/spline_energy.py"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
