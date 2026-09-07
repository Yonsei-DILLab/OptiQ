"""CPU-sized regression: sparse diagnostics must not change training state."""
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir

from common.type_aliases import ReplayBufferSamplesNp
from optiq_dime.algorithm import OptiQDIME, _pack_metrics
from optiq_dime.policy import OptiQPolicy


def configuration():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        return compose(config_name="parameter_sweep", overrides=[
            "alg.critic.hs=[16,16]", "alg.critic.n_atoms=11",
            "alg.actor.hidden_dims=[16,16]", "wandb.activate=false",
        ])


def states():
    cfg = configuration()
    policy = OptiQPolicy(spaces.Box(-1, 1, (3,)), spaces.Box(-1, 1, (2,)), cfg)
    policy.build(jax.random.PRNGKey(10), lambda _: 3e-4, 3e-4)
    return policy


def assert_trees_equal(first, second):
    # Different XLA fusion choices may differ at floating-point roundoff.
    for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("mode", ["stratified", "exact", "skewed"])
@pytest.mark.parametrize("anchor", [False, True])
def test_actor_diagnostics_do_not_change_update(mode, anchor):
    policy = states()
    kwargs = dict(
        actor_state=policy.actor_state, qf_state=policy.qf_state,
        observations=jnp.ones((2, 3)), key=jax.random.PRNGKey(20),
        z_atoms=jnp.linspace(-200, 200, 11), num_policy_samples=4,
        proposals_per_policy_sample=5 if anchor else 4,
        proposal_sampling_mode=mode, proposal_std=0.2, proposal_clip=0.5,
        include_anchor=anchor, density_correction=True, density_beta=0.1,
        adaptive_density_beta=False, minimum_source_ess=4.0,
        density_beta_grid_size=17, temperature=0.25,
        sinkhorn_epsilon=0.05, sinkhorn_iterations=5,
        source_q_eval="mean", transport_target_mode="argmax",
    )
    full = OptiQDIME.update_actor(**kwargs)
    sparse = OptiQDIME.update_actor(**kwargs, collect_metrics=False)
    assert full[3] and sparse[3] == {}
    assert_trees_equal(full[:3], sparse[:3])
    np.testing.assert_array_equal(full[2], sparse[2])
    packed = np.asarray(_pack_metrics(full[3]))
    np.testing.assert_allclose(packed, [float(full[3][k]) for k in sorted(full[3])])


@pytest.mark.parametrize("adaptive,adaptive_mode", [(False, "ess"), (True, "ess"), (True, "kl")])
def test_full_utd_update_is_preserved(adaptive, adaptive_mode):
    policy = states()
    data = ReplayBufferSamplesNp(
        jnp.ones((4, 3)), jnp.zeros((4, 2)), jnp.ones((4, 3)) * 0.9,
        jnp.zeros(4), jnp.ones(4),
    )
    kwargs = dict(
        crossq_style=True, use_bnstats_from_live_net=False, gamma=0.99,
        tau=1.0, policy_tau=1.0, gradient_steps=2, data=data,
        policy_delay_indices=flax.core.FrozenDict({0: True, 1: True}),
        qf_state=policy.qf_state, actor_state=policy.actor_state,
        target_actor_state=policy.target_actor_state, ent_coef_state=None,
        key=jax.random.PRNGKey(30), n_env_interacts=6000,
        v_min=-200, v_max=200, entr_coeff=0.005, num_atoms=11,
        num_policy_samples=4, proposals_per_policy_sample=5,
        proposal_sampling_mode="stratified", proposal_std=0.2, proposal_clip=0.5,
        include_anchor=True, density_correction=True, density_beta=0.1,
        adaptive_density_beta=adaptive, minimum_source_ess=4.0,
        density_beta_grid_size=17, temperature=0.25, sinkhorn_epsilon=0.05,
        sinkhorn_iterations=5, source_q_eval="mean", transport_target_mode="argmax",
        td_noise_std=0.2, td_noise_clip=0.5,
        adaptive_density_beta_mode=adaptive_mode,
    )
    full = OptiQDIME._train(**kwargs)
    sparse = OptiQDIME._train(**kwargs, collect_metrics=False)
    assert full[-1] and sparse[-1] == {}
    assert_trees_equal(full[:-1], sparse[:-1])
    np.testing.assert_array_equal(full[-2], sparse[-2])
    assert np.asarray(_pack_metrics(full[-1])).shape == (len(full[-1]),)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("target_mode", ["argmax", "barycentric"])
def test_kl_mode_and_anchor_diagnostics(anchor, target_mode):
    policy = states()
    repeats = 5 if anchor else 4
    kwargs = dict(
        actor_state=policy.actor_state, qf_state=policy.qf_state,
        observations=jnp.ones((2, 3)), key=jax.random.PRNGKey(40),
        z_atoms=jnp.linspace(-200, 200, 11), num_policy_samples=4,
        proposals_per_policy_sample=repeats,
        proposal_sampling_mode="stratified", proposal_std=0.2, proposal_clip=0.5,
        include_anchor=anchor, density_correction=True, density_beta=0.1,
        adaptive_density_beta=True, adaptive_density_beta_mode="kl",
        minimum_source_ess=4.0, density_beta_grid_size=17, temperature=0.25,
        sinkhorn_epsilon=0.05, sinkhorn_iterations=5,
        source_q_eval="mean", transport_target_mode=target_mode,
    )
    full = OptiQDIME.update_actor(**kwargs)
    sparse = OptiQDIME.update_actor(**kwargs, collect_metrics=False)
    assert_trees_equal(full[:3], sparse[:3])
    np.testing.assert_array_equal(full[2], sparse[2])
    assert sparse[3] == {}
    metrics = {key: float(value) for key, value in full[3].items()}
    assert np.isfinite(list(metrics.values())).all()
    assert metrics["density_beta_kl_valid"] == 1.0
    assert 0.0 <= metrics["density_beta_min"] <= metrics["density_beta_max"] <= 1.0
    assert metrics["density_beta_kl"] <= metrics["density_beta_kl_budget"] + 1e-6
    assert 0.0 <= metrics["density_beta_kl_utilization"] <= 1.0 + 1e-5
    assert metrics["candidate_count"] == 4 * repeats
    assert metrics["anchor_present"] == float(anchor)
    assert metrics["anchor_selection_valid"] == float(anchor and target_mode == "argmax")
    for name in ("anchor_target_mass", "anchor_selected_fraction", "own_anchor_selected_fraction"):
        assert 0.0 <= metrics[name] <= 1.0
        if not anchor:
            assert metrics[name] == 0.0
    if target_mode == "barycentric":
        assert metrics["anchor_selected_fraction"] == 0.0
        assert metrics["own_anchor_selected_fraction"] == 0.0
    assert metrics["own_anchor_selected_fraction"] <= metrics["anchor_selected_fraction"]


def test_sweep_configuration():
    cfg = configuration()
    assert cfg.eval_interval == 10000
    assert cfg.diagnostics_interval == 1000
    assert cfg.alg.utd == 2
    assert cfg.alg.actor.proposal_sampling_mode == "stratified"
    assert cfg.alg.actor.adaptive_density_beta_mode == "ess"
    assert cfg.checkpoint_interval == 50000


def test_host_training_loop(tmp_path):
    from run_optiq_dime import create_algorithm

    cfg = configuration()
    cfg.env_name = "Pendulum-v1"
    cfg.task = "cpu-smoke"
    cfg.output_root = str(tmp_path)
    cfg.eval_interval = 0
    cfg.checkpoint_interval = 0
    cfg.diagnostics_interval = 4
    cfg.alg.learning_starts = 4
    cfg.alg.buffer_size = 64
    cfg.alg.batch_size = 2
    cfg.alg.actor.num_policy_samples = 4
    cfg.alg.actor.proposals_per_policy_sample = 5
    model, callbacks = create_algorithm(cfg)
    try:
        model.learn(total_timesteps=12, callback=callbacks, progress_bar=False)
        assert model._n_updates == 16
        assert model._last_diagnostics_bucket == 3
    finally:
        model.env.close()
        callbacks.callbacks[0].eval_env.close()
