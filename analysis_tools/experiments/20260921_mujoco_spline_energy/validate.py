"""Cheap checks for normalization, sampling, gradients and MuJoCo shapes."""
import hashlib
import json
from pathlib import Path

from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from circuit import (ConditionalSplineCircuit, coarse_leaf_probabilities,
                     log_prob_from_output, q_from_output, sample_action)
from train import ENVS, make_update, to_environment_action, to_normalized_action


def main():
    model = ConditionalSplineCircuit(action_dim=1, rank=4, knots=17, hidden_dims=(32, 32))
    obs = jnp.asarray([[0.2, -0.3], [-0.1, 0.7]], jnp.float32)
    params = model.init(jax.random.PRNGKey(0), obs)["params"]
    output = model.apply({"params": params}, obs)
    leaf_cells = coarse_leaf_probabilities(output, cells=8)
    np.testing.assert_allclose(np.asarray(leaf_cells.sum(-1)), 1., rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(jnp.exp(output["log_weights"]).sum(-1)), 1., rtol=2e-6)
    repeated = jnp.repeat(obs[:1], 65536, axis=0)
    samples = np.asarray(sample_action(params, repeated, jax.random.PRNGKey(11), model=model))[:, 0]
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) <= 1.
    empirical = np.histogram(samples, bins=np.linspace(-1, 1, 9))[0] / len(samples)
    expected = np.asarray(jnp.einsum("r,rc->c", jnp.exp(output["log_weights"][0]), leaf_cells[0, :, 0]))
    assert np.max(np.abs(empirical - expected)) < .012, (empirical, expected)
    actions = jnp.asarray([[0.1], [-0.4]], jnp.float32)
    logp = log_prob_from_output(output, actions)
    q = q_from_output(output, actions, .25)
    np.testing.assert_allclose(np.asarray(q), np.asarray(output["value"] + .25 * logp), rtol=1e-6)
    state = TrainState.create(apply_fn=model.apply, params=params,
        tx=optax.chain(optax.clip_by_global_norm(10.), optax.adam(3e-4)))
    update = make_update(model, .25, .99, .005, 10.)
    batch = {"obs": obs, "actions": actions, "rewards": jnp.asarray([1., -.2]),
             "next_obs": obs[::-1], "not_terminal": jnp.asarray([1., 0.])}
    new_state, target, metrics = update(state, params, batch)
    assert int(new_state.step) == 1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((new_state.params, target, metrics)))
    delta = max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(new_state.params)))
    assert delta > 0.
    environments = {}
    for name in ENVS:
        env = gym.make(name)
        try:
            observation, _ = env.reset(seed=0)
            assert np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all()
            probe = np.linspace(-1., 1., env.action_space.shape[0], dtype=np.float32)
            physical = to_environment_action(probe, env.action_space.low, env.action_space.high)
            np.testing.assert_allclose(to_normalized_action(
                physical, env.action_space.low, env.action_space.high), probe, atol=2e-6)
            environments[name] = {"observation_shape": list(observation.shape),
                                  "action_shape": list(env.action_space.shape),
                                  "action_low": env.action_space.low.tolist(),
                                  "action_high": env.action_space.high.tolist(),
                                  "horizon": env.spec.max_episode_steps}
        finally:
            env.close()
    humanoid = ConditionalSplineCircuit(action_dim=17)
    hp = humanoid.init(jax.random.PRNGKey(3), jnp.zeros((2, 376), jnp.float32))["params"]
    count = sum(x.size for x in jax.tree_util.tree_leaves(hp))
    record = {"passed": True, "max_histogram_error": float(np.max(np.abs(empirical-expected))),
              "gradient_parameter_delta": delta, "humanoid_parameters": count,
              "environments": environments,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py")}}
    destination = Path(__file__).resolve().parents[4] / "checks" / "spline_energy_cpu_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
