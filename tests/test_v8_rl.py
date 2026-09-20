"""The v8 RL route shares the validated GMM actor core and soft TD path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optiq_dime.algorithm import OptiQDIME
import optiq_dime.algorithm as algorithm
from run_optiq_dime import validate_config, v8_algorithm_metadata
from scripts.verify_v8 import verify
from test_v7_rl import actor_arguments, actor_state, linear_critic_state


def arguments():
    args = actor_arguments(actor_state(), linear_critic_state(), aggregation="min", sites=16, pairs=4)
    args.update(distillation_loss="ot_latent_conditional_sac", ot_student_action="latent",
                ot_potential_mode="fresh_sinkhorn", sinkhorn_epsilon=.1, sinkhorn_iterations=2000,
                proposals_per_policy_sample=1, proposal_sampling_mode="stratified",
                ot_min_iterations=5, ot_relative_tolerance=1e-3)
    return args


def test_configuration_is_distinct_and_preserves_soft_td():
    cfg = verify(["benchmark=ant"])
    metadata = v8_algorithm_metadata(cfg)
    assert cfg.alg.actor.ot_num_latents == 4096
    assert cfg.alg.actor.teacher_proposal_components == 256
    assert cfg.alg.actor.ot_teacher_resample_count == cfg.alg.actor.num_policy_samples == 16
    assert cfg.alg.actor.source_q_eval == "min"
    assert cfg.alg.critic.backup_mode == "soft_td"
    assert cfg.alg.actor.entropy_samples == 16
    assert not metadata["ot_dual_state_persistent"]
    assert not metadata["actor_source_importance_correction"]
    assert cfg.alg.actor.sinkhorn_epsilon == .1
    assert cfg.alg.ent_coef.init == cfg.alg.actor.temperature


@pytest.mark.parametrize("override", ["alg.actor.ot_potential_mode=persistent_dual",
                                      "alg.actor.sinkhorn_epsilon=0",
                                      "alg.actor.source_q_eval=mean",
                                      "alg.actor.ot_relative_tolerance=0",
                                      "alg.actor.ot_min_iterations=1000001"])
def test_incompatible_v8_configuration_fails(override):
    with pytest.raises(ValueError):
        verify([override])


def test_route_uses_current_min_q_with_action_gradient_and_per_state_outputs(monkeypatch):
    captured = {}
    actions = jnp.array([[[.1, -.2], [.3, .4]], [[-.1, .2], [.4, -.3]]])
    def inspect_core(state, obs, key, q_fn, **settings):
        captured.update(settings)
        return state, q_fn(obs, actions).sum(), key, {"actor_update_accepted": jnp.asarray(1.), "q": q_fn(obs, actions),
            "dq": jax.grad(lambda a: q_fn(obs, a).sum())(actions)}
    monkeypatch.setattr(algorithm, "update_v8_conditional_sac_actor", inspect_core)
    args = arguments()
    result = OptiQDIME.update_actor.__wrapped__(**args)
    p = args["qf_state"].params
    heads = np.einsum("hd,bkd->hbk", p["weights"], actions) + np.asarray(p["bias"])[:, None, None]
    np.testing.assert_allclose(result[3]["q"], heads.min(0))
    expected = np.asarray(p["weights"])[heads.argmin(0)]
    np.testing.assert_allclose(result[3]["dq"], expected)
    assert captured["max_iterations"] == 2000
    assert captured["relative_tolerance"] == 1e-3
    assert captured["epsilon"] == .1
    assert "source_potential" not in captured and "dual_state" not in captured


def test_actual_rl_actor_update_is_finite_and_has_no_persistent_state():
    args = arguments()
    updated, loss, key, metrics = OptiQDIME.update_actor(**args)
    jax.block_until_ready(loss)
    assert int(updated.step) == 1
    assert np.isfinite(loss)
    assert float(metrics["actor_update_accepted"]) == 1.
    assert float(metrics["actor_source_importance_used"]) == 0.
    assert float(metrics["actor_nll_used"]) == 0.
    assert not np.array_equal(key, args["key"])


def test_rejected_core_update_preserves_rl_entry_rng(monkeypatch):
    def reject(state, obs, key, q_fn, **settings):
        return state, jnp.asarray(0.), key, {"actor_update_accepted": jnp.asarray(0.)}
    monkeypatch.setattr(algorithm, "update_v8_conditional_sac_actor", reject)
    args = arguments()
    updated, _, key, _ = OptiQDIME.update_actor.__wrapped__(**args)
    np.testing.assert_array_equal(key, args["key"])
    for left, right in zip(jax.tree.leaves(updated), jax.tree.leaves(args["actor_state"])):
        np.testing.assert_array_equal(left, right)


def test_ant_replay_to_actor_soft_td_and_checkpoint_integration(tmp_path):
    """Six CPU environment steps validate the route; this is not an RL experiment."""
    from pathlib import Path
    from flax import serialization
    from stable_baselines3.common.logger import configure
    from run_optiq_dime import create_algorithm
    cfg = verify([
        "benchmark=ant", f"output_root={tmp_path}", "alg.batch_size=2", "alg.buffer_size=32",
        "alg.learning_starts=2", "alg.actor.learning_starts=2", "num_eval_episodes=1",
        "eval_interval=3", "diagnostic_interval=3", "checkpoint_interval=3",
    ])
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
    callback = callbacks.callbacks[0]
    callback.eval_env.envs[0].env._max_episode_steps = 2
    model.get_env().envs[0].env._max_episode_steps = 2
    try:
        model.learn(total_timesteps=6, callback=callbacks)
        assert model.dual_state is None
        assert model.backup_mode == "soft_td"
        assert int(model.policy.actor_state.step) == int(model.policy.qf_state.step) == 4
        for state, prefix in ((model.policy.actor_state, "actor"), (model.policy.qf_state, "critic")):
            path = Path(model.model_save_path) / f"{prefix}_state_6.msgpack"
            restored = serialization.from_bytes(state, path.read_bytes())
            for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
                np.testing.assert_array_equal(left, right)
        assert not list(Path(model.model_save_path).glob("dual_state*"))
    finally:
        callback.eval_env.close()
        model.get_env().close()
        model.logger.close()
        OptiQDIME._train.clear_cache()
