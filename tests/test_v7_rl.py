"""v7 RL routing and a bounded Ant integration validation, not a training run."""
import copy
import csv
import inspect
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from stable_baselines3.common.logger import configure

from common.type_aliases import RLTrainState
import optiq_dime.algorithm as algorithm
from optiq_dime.algorithm import OptiQDIME
import optiq_dime.policy as policy_module
from optiq_dime.policy import OptiQPolicy
from optiq_dime.semi_implicit import idac_action_and_log_density
from optiq_dime.persistent_transport import create_dual_state
from run_optiq_dime import create_algorithm
from scripts.verify_v7 import verify
from test_semi_implicit import actor_state, critic_state


def actor_arguments(actor, critic, *, aggregation="mean", sites=32, pairs=4):
    return dict(
        actor_state=actor, qf_state=critic, observations=jnp.ones((2, 3)),
        key=jax.random.PRNGKey(73), z_atoms=jnp.array([-3600.]),
        num_policy_samples=pairs, proposals_per_policy_sample=4,
        proposal_sampling_mode="exact", proposal_std=.05, proposal_clip=.5,
        include_anchor=False, density_correction=True, density_beta=1.,
        adaptive_density_beta=False, minimum_source_ess=16., density_beta_grid_size=257,
        temperature=.25, sinkhorn_epsilon=.1, sinkhorn_iterations=100,
        source_q_eval=aggregation, transport_target_mode="argmax", semi_implicit=True,
        normalize_ot_cost=False, distillation_loss="ot_conditional_sac",
        teacher_distribution="conditional_mixture", soft_proximal_ess_fraction=0.,
        entropy_diagnostics=False, ot_student_action="latent", ot_num_latents=sites,
        ot_teacher_resample_count=pairs, ot_latent_seed=19, teacher_proposal_components=pairs,
    )


def linear_critic(variables, observations, actions, rngs=None, mutable=False, train=False):
    params = variables["params"]
    values = jnp.einsum("hd,bd->hb", params["weights"], actions) + params["bias"][:, None]
    result = values[..., None]
    return (result, {"batch_stats": {}}) if mutable else result


def linear_critic_state():
    params = {"weights": jnp.array([[1., 2.], [-3., 4.]]), "bias": jnp.array([1., 3.])}
    return RLTrainState.create(
        apply_fn=linear_critic, params=params, batch_stats={},
        target_params=jax.tree.map(lambda value: value + 100., params),
        target_batch_stats={}, tx=optax.sgd(.01),
    )


@pytest.mark.parametrize("aggregation", ["mean", "min"])
def test_v7_route_uses_current_critic_for_teacher_and_actor_with_live_action_gradient(monkeypatch, aggregation):
    """Probe the actual RL Q closure; target critics must never enter the actor."""
    actor, critic = actor_state(), linear_critic_state()
    teacher_actions = jnp.array([[[.1, -.2], [.4, .3]], [[-.3, .7], [.8, -.6]]])
    actor_actions = -teacher_actions
    captured = {}

    def inspect_core(state, observations, key, q_fn, **kwargs):
        captured.update(kwargs)
        teacher_q = q_fn(observations, teacher_actions)
        actor_q = q_fn(observations, actor_actions)
        action_gradient = jax.grad(lambda actions: q_fn(observations, actions).sum())(actor_actions)
        return state, actor_q.sum(), key, {
            "teacher_q": teacher_q, "actor_q": actor_q, "action_gradient": action_gradient,
        }

    monkeypatch.setattr(algorithm, "update_conditional_sac_actor", inspect_core)
    args = actor_arguments(actor, critic, aggregation=aggregation, sites=4096, pairs=16)
    args["teacher_proposal_components"] = 256
    args["proposals_per_policy_sample"] = 1
    args["proposal_sampling_mode"] = "stratified"
    # The Python routing body exposes the closure without a cached prior trace.
    route = OptiQDIME.update_actor.__wrapped__
    result = route(**args)
    params = critic.params

    def oracle(actions):
        heads = np.einsum("hd,bkd->hbk", np.asarray(params["weights"]), np.asarray(actions))
        heads += np.asarray(params["bias"])[:, None, None]
        return heads.mean(axis=0) if aggregation == "mean" else heads.min(axis=0)

    np.testing.assert_allclose(result[3]["teacher_q"], oracle(teacher_actions), atol=1e-6)
    np.testing.assert_allclose(result[3]["actor_q"], oracle(actor_actions), atol=1e-6)
    if aggregation == "mean":
        expected_gradient = np.broadcast_to(np.asarray(params["weights"]).mean(axis=0), actor_actions.shape)
    else:
        heads = np.einsum("hd,bkd->hbk", np.asarray(params["weights"]), np.asarray(actor_actions))
        heads += np.asarray(params["bias"])[:, None, None]
        expected_gradient = np.asarray(params["weights"])[heads.argmin(axis=0)]
    np.testing.assert_allclose(result[3]["action_gradient"], expected_gradient, atol=1e-6)
    frozen_critic_gradient = jax.grad(
        lambda p: route(**dict(args, qf_state=critic.replace(params=p)))[1]
    )(critic.params)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(frozen_critic_gradient))
    assert captured["num_students"] == 4096
    assert captured["proposal_components"] == 256 and captured["proposals_per_component"] == 1
    assert captured["teacher_sampling_mode"] == "stratified"
    assert captured["actor_samples"] == captured["teacher_resample_count"] == 16
    assert captured["latent_seed"] == 19


def test_v7_route_performs_a_finite_actor_update_and_ignores_target_critic():
    """A tiny quadrature validates the real actor path without an environment."""
    actor, critic = actor_state(), linear_critic_state()
    args = actor_arguments(actor, critic)
    updated, loss, key, metrics = OptiQDIME.update_actor(**args)
    jax.block_until_ready(loss)
    assert int(updated.step) == 1 and np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    for head in ("mu", "log_std"):
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(actor.params[head]), jax.tree.leaves(updated.params[head])))
    assert float(metrics["v7_conditional_sac_used"]) == 1.
    assert float(metrics["actor_nll_used"]) == 0.
    assert float(metrics["ot_source_count"]) == 32.
    assert float(metrics["teacher_candidate_count"]) == 16.
    assert float(metrics["ot_teacher_count"]) == float(metrics["actor_training_pairs"]) == 4.
    changed_target = critic.replace(target_params=jax.tree.map(lambda x: x - 200., critic.target_params))
    same = OptiQDIME.update_actor(**dict(args, qf_state=changed_target))
    for left, right in zip(jax.tree.leaves((updated, loss, key, metrics)), jax.tree.leaves(same)):
        np.testing.assert_array_equal(left, right)


def forbidden_entropy_or_guard(*args, **kwargs):
    raise AssertionError("This test forbids the selected entropy or soft-guard helper")


def test_persistent_route_keeps_dual_optimizer_and_freezes_dual_in_actor_loss():
    actor, critic = actor_state(), linear_critic_state()
    args = actor_arguments(actor, critic)
    dual = create_dual_state(jax.random.PRNGKey(102), args["observations"],
                             num_sources=32, hidden_dims=(16, 16), learning_rate=1e-4)
    args.update(dual_state=dual, ot_potential_mode="persistent_dual")
    updated, updated_dual, loss, key, metrics = OptiQDIME.update_actor(**args)
    jax.block_until_ready(loss)
    assert int(updated.step) == int(updated_dual.step) == 1
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(dual.params), jax.tree.leaves(updated_dual.params)))
    again = OptiQDIME.update_actor(**dict(args, actor_state=updated, dual_state=updated_dual, key=key))
    assert int(again[0].step) == int(again[1].step) == 2
    # Actor scalar is independent of critic and dual PARAMETERS, while core
    # tests verify its live gradients through action inputs Q(a) and Pr(i|a).
    for state_name, state in (("qf_state", critic), ("dual_state", dual)):
        gradients = jax.grad(lambda params: OptiQDIME.update_actor(**dict(
            args, **{state_name: state.replace(params=params)}))[2])(state.params)
        assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(gradients))
    assert float(metrics["ot_fresh_solve"]) == 0.


def test_persistent_route_requires_a_dual_state():
    with pytest.raises(ValueError, match="dual TrainState"):
        OptiQDIME.update_actor(**dict(actor_arguments(actor_state(), linear_critic_state()),
                                     ot_potential_mode="persistent_dual"))


def test_legacy_plain_td_backup_has_no_actor_temperature_bonus(monkeypatch):
    monkeypatch.setattr(algorithm, "idac_action_and_log_density", forbidden_entropy_or_guard)
    actor, critic = actor_state(), critic_state()
    rewards, dones = jnp.array([1., 3.]), jnp.array([0., 1.])
    args = dict(
        crossq_style=False, use_bnstats_from_live_net=False, gamma=.9,
        target_actor_state=actor, qf_state=critic, observations=jnp.ones((2, 3)),
        actions=jnp.zeros((2, 2)), next_observations=jnp.ones((2, 3)),
        rewards=rewards, dones=dones, num_atoms=1, z_atoms=jnp.array([-3600.]),
        v_min=-3600, v_max=3600, entr_coeff=0., td_noise_std=0., td_noise_clip=0.,
        key=jax.random.PRNGKey(92), semi_implicit=True, entropy_samples=0,
        temperature=.25, backup_mode="td",
    )
    updated, metrics, key = OptiQDIME.update_critic(**args)
    target = np.asarray(rewards) + (1 - np.asarray(dones)) * .9 * 10.
    np.testing.assert_allclose(metrics["next_q_values"], target.mean(), atol=1e-6)
    expected_loss = ((np.array([2., 6.])[:, None] - target[None]) ** 2).mean(axis=1).sum()
    np.testing.assert_allclose(metrics["critic_loss"], expected_loss, atol=1e-6)
    for metric in ("ent_coef", "backup_entropy_term", "backup_discounted_entropy_term"):
        assert float(metrics[metric]) == 0.
    different_temperature = OptiQDIME.update_critic(**dict(args, temperature=1.))
    for left, right in zip(jax.tree.leaves((updated, metrics, key)), jax.tree.leaves(different_temperature)):
        np.testing.assert_array_equal(left, right)


def test_v7_soft_td_uses_marginal_mixture_entropy_and_stops_target_gradient():
    actor, critic = actor_state(), critic_state()
    rewards, dones = jnp.array([1., 3.]), jnp.array([0., 1.])
    observations = jnp.ones((2, 3))
    key = jax.random.PRNGKey(92)
    args = dict(
        crossq_style=False, use_bnstats_from_live_net=False, gamma=.9,
        target_actor_state=actor, qf_state=critic, observations=observations,
        actions=jnp.zeros((2, 2)), next_observations=observations,
        rewards=rewards, dones=dones, num_atoms=1, z_atoms=jnp.array([-3600.]),
        v_min=-3600, v_max=3600, entr_coeff=0., td_noise_std=0., td_noise_clip=0.,
        key=key, semi_implicit=True, entropy_samples=16, temperature=.25, backup_mode="soft_td",
    )
    actor_key = jax.random.split(key, 6)[1]
    _, log_mix = idac_action_and_log_density(actor, observations, actor_key, 16)
    _, metrics, _ = OptiQDIME.update_critic(**args)
    entropy = -.25 * np.asarray(log_mix)
    target = np.asarray(rewards) + (1 - np.asarray(dones)) * .9 * (10. + entropy)
    np.testing.assert_allclose(metrics["next_q_values"], target.mean(), atol=1e-6)
    expected_loss = ((np.array([2., 6.])[:, None] - target[None]) ** 2).mean(axis=1).sum()
    np.testing.assert_allclose(metrics["critic_loss"], expected_loss, rtol=2e-6)
    np.testing.assert_allclose(metrics["backup_entropy_term"], entropy.mean(), atol=1e-6)
    assert float(metrics["ent_coef"]) == .25
    target_actor_gradient = jax.grad(lambda p: OptiQDIME.update_critic(**dict(
        args, target_actor_state=actor.replace(params=p)))[1]["critic_loss"])(actor.params)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(target_actor_gradient))
    _, warmer, _ = OptiQDIME.update_critic(**dict(args, temperature=1.))
    target_warmer = np.asarray(rewards) + (1 - np.asarray(dones)) * .9 * (10. - np.asarray(log_mix))
    np.testing.assert_allclose(warmer["next_q_values"], target_warmer.mean(), atol=1e-6)
    assert not np.isclose(float(warmer["next_q_values"]), float(metrics["next_q_values"]))


def adam_states(tree):
    if hasattr(tree, "mu") and hasattr(tree, "nu") and hasattr(tree, "count"):
        yield tree
    elif isinstance(tree, (tuple, list)):
        for part in tree:
            yield from adam_states(part)
    elif isinstance(tree, dict):
        for part in tree.values():
            yield from adam_states(part)


def test_v7_actual_ant_full_quadrature_soft_td_evaluation_and_checkpoints(tmp_path, monkeypatch):
    """Only 8 env steps: integration validation after the GMM-first check.

    Warmup, replay size, batch and evaluation horizon are validation overrides.
    H4096/P256/M256/K16, networks, LR, UTD and actor frequency stay canonical.
    """
    cfg = verify([
        "benchmark=ant", f"output_root={tmp_path}", "alg.batch_size=2", "alg.buffer_size=32",
        "alg.learning_starts=2", "alg.actor.learning_starts=2", "num_eval_episodes=2",
        "eval_interval=4", "diagnostic_interval=4", "checkpoint_interval=4",
    ])
    assert cfg.alg.utd == cfg.alg.policy_delay == 1
    assert cfg.alg.optimizer.lr_actor == cfg.alg.optimizer.lr_critic == .0003
    for name in ("conditional_mixture_log_prob", "sampled_soft_update"):
        monkeypatch.setattr(algorithm, name, forbidden_entropy_or_guard)
    entropy_counts = []
    original_entropy = algorithm.idac_action_and_log_density

    def record_entropy(actor, observations, key, count):
        entropy_counts.append(count)
        return original_entropy(actor, observations, key, count)

    monkeypatch.setattr(algorithm, "idac_action_and_log_density", record_entropy)
    for function in (OptiQDIME._train, OptiQDIME.update_critic, OptiQDIME.update_actor):
        function.clear_cache()

    seen, optimizer_limits = [], []
    original_optimizer = policy_module.adam_with_grad_clip

    def record_optimizer(*args, **kwargs):
        optimizer_limits.append(kwargs["max_grad_norm"])
        return original_optimizer(*args, **kwargs)

    monkeypatch.setattr(policy_module, "adam_with_grad_clip", record_optimizer)
    original_actor = OptiQDIME.update_actor
    signature = inspect.signature(original_actor)

    def record_actor(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        seen.append({name: arguments[name] for name in (
            "distillation_loss", "ot_student_action", "ot_num_latents",
            "ot_teacher_resample_count", "ot_latent_seed", "num_policy_samples",
            "proposals_per_policy_sample", "source_q_eval", "teacher_proposal_components",
            "proposal_sampling_mode",
            "ot_potential_mode",
        )})
        return original_actor(*args, **kwargs)

    monkeypatch.setattr(OptiQDIME, "update_actor", staticmethod(record_actor))
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path / "validation_logs"), ["csv"]))
    callback = callbacks.callbacks[0]
    callback.eval_env.envs[0].env._max_episode_steps = 2
    model.get_env().envs[0].env._max_episode_steps = 2
    before = copy.deepcopy(model.policy.actor_state.params)
    initial_dual = model.dual_state
    try:
        model.learn(total_timesteps=8, callback=callbacks)
        assert seen and all(item == {
            "distillation_loss": "ot_conditional_sac", "ot_student_action": "latent",
            "ot_num_latents": 4096, "ot_teacher_resample_count": 16, "ot_latent_seed": 0,
            "num_policy_samples": 16, "proposals_per_policy_sample": 1, "source_q_eval": "mean",
            "teacher_proposal_components": 256, "proposal_sampling_mode": "stratified",
            "ot_potential_mode": "persistent_dual",
        } for item in seen)
        assert optimizer_limits == [None, None]
        assert model._n_updates == int(model.policy.actor_state.step) == int(model.policy.qf_state.step) == 6
        assert int(model.dual_state.step) == 6
        assert model.backup_mode == "soft_td" and model.behavior_uniform_count == 0
        assert entropy_counts and all(count == 16 for count in entropy_counts)
        assert model.soft_guard_attempts == model.soft_guard_accepts == 0
        for head in ("mu", "log_std"):
            assert not np.array_equal(before[head]["kernel"], model.policy.actor_state.params[head]["kernel"])
        for state, prefix in ((model.policy.actor_state, "actor"), (model.policy.qf_state, "critic"),
                              (model.dual_state, "dual")):
            moments = list(adam_states(state.opt_state))
            assert len(moments) == 1 and int(moments[0].count) == 6
            assert any(np.count_nonzero(x) for x in jax.tree.leaves(moments[0].mu))
            assert all(np.isfinite(x).all() for x in jax.tree.leaves(moments[0].nu))
            path = Path(model.model_save_path) / f"{prefix}_state_8.msgpack"
            restored = serialization.from_bytes(state, path.read_bytes())
            for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
                np.testing.assert_array_equal(left, right)
        saved_dual = model.dual_state
        model.dual_state = initial_dual
        model.load_model(model.model_save_path, 8, 8)
        for left, right in zip(jax.tree.leaves(saved_dual), jax.tree.leaves(model.dual_state)):
            np.testing.assert_array_equal(left, right)
        with pytest.raises(FileNotFoundError, match="Persistent-dual checkpoint requires"):
            model.load_model(tmp_path / "missing_checkpoint", 8, 8)
        obs = jnp.zeros((1, model.observation_space.shape[0]))
        restored_actor = serialization.from_bytes(
            model.policy.actor_state, (Path(model.model_save_path) / "actor_state_8.msgpack").read_bytes())
        np.testing.assert_array_equal(
            OptiQPolicy.sample_action(restored_actor, obs, jax.random.PRNGKey(12)),
            OptiQPolicy.sample_action(model.policy.actor_state, obs, jax.random.PRNGKey(12)),
        )
        assert callback.evaluations_timesteps == [1, 4, 8]
        for mode in callback.MODES:
            with np.load(callback.directory / f"evaluations_{mode}.npz") as data:
                assert data["results"].shape == (3, 2)
                assert np.isfinite(data["results"]).all()
        for seed_kind in ("env_seeds", "policy_seeds"):
            np.testing.assert_array_equal(callback.histories["zero_z"][seed_kind],
                                          callback.histories["stochastic_z"][seed_kind])
        with (tmp_path / "validation_logs/progress.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        diagnostics = [row for row in rows if row.get("train/v7_conditional_sac_used")]
        assert diagnostics
        for row in diagnostics:
            assert float(row["train/v7_conditional_sac_used"]) == 1.
            assert float(row["train/actor_nll_used"]) == 0.
            assert float(row["train/ot_source_count"]) == 4096.
            assert float(row["train/teacher_candidate_count"]) == 256.
            assert float(row["train/teacher_proposal_component_count"]) == 256.
            assert float(row["train/teacher_one_per_latent"]) == 1.
            assert float(row["train/ot_teacher_count"]) == float(row["train/actor_training_pairs"]) == 16.
            assert float(row["train/actor_entropy_temperature"]) == .25
            assert float(row["train/ot_fresh_solve"]) == 0.
            assert float(row["train/ent_coef"]) == .25
            np.testing.assert_allclose(float(row["train/backup_entropy_term"]),
                                       .25 * float(row["train/backup_entropy_lower"]), rtol=1e-5)
        assert any(abs(float(row["train/backup_entropy_term"])) > 1e-8 for row in diagnostics)
        assert model.replay_buffer.timeouts[:8].sum() > 0
        assert not np.asarray(model.replay_buffer._get_samples(np.arange(8)).dones).any()
    finally:
        callback.eval_env.close()
        model.get_env().close()
        model.logger.close()
        OptiQDIME._train.clear_cache()
