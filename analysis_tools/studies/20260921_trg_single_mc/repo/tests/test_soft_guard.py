import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from common.type_aliases import RLTrainState
from optiq_dime.soft_improvement import sampled_soft_update


def gaussian(variables, observations, latents):
    p = variables["params"]
    return (jnp.broadcast_to(p["mu"]["bias"], latents.shape),
            jnp.broadcast_to(p["ls"], latents.shape))


def flat_q(variables, observations, actions, **kwargs):
    return jnp.broadcast_to(variables["params"]["q"][:, None, None], (2, len(actions), 1))


def setup():
    old = TrainState.create(apply_fn=gaussian,
        params={"mu": {"bias": jnp.zeros(2)}, "ls": jnp.full((2,), jnp.log(.5))},
        tx=optax.adam(.01))
    q = RLTrainState.create(apply_fn=flat_q, params={"q": jnp.zeros(2)},
        target_params={"q": jnp.zeros(2)}, batch_stats={}, target_batch_stats={}, tx=optax.sgd(.01))
    return old, q


@pytest.mark.parametrize("sigma,accepted", [(.9, True), (.1, False), (.5, False), (float("nan"), False)])
def test_soft_filter_accepts_entropy_gain_and_rolls_back_full_optimizer(sigma, accepted):
    old, q = setup()
    candidate = old.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, old.params))
    candidate = candidate.replace(params={"mu": {"bias": jnp.zeros(2)},
                                          "ls": jnp.full((2,), jnp.log(sigma))})
    selected, metrics = jax.jit(sampled_soft_update, static_argnames=("components", "draws"))(
        old, candidate, q, jnp.zeros((128, 3)), jax.random.PRNGKey(9), .5,
        jnp.zeros(1), components=4, draws=32)
    assert bool(metrics["soft_guard_accepted"]) == accepted
    expected = candidate if accepted else old
    for actual, wanted in zip(jax.tree_util.tree_leaves(selected), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(actual, wanted)
    assert int(selected.step) == int(accepted)


@pytest.mark.parametrize("override", ["alg.actor.soft_guard.batch_size=1",
    "alg.actor.soft_guard.draws=1", "alg.actor.soft_guard.components=0",
    "alg.actor.soft_guard.standard_error_multiplier=-1"])
def test_invalid_guard_config_rejected(override):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from run_optiq_dime import validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="mujoco_v2_guarded", overrides=[override])
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("standard_error_multiplier", [0.0, 2.0])
def test_guard_in_common_humanoid_loop(tmp_path, standard_error_multiplier):
    import csv
    import gymnasium as gym
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from stable_baselines3.common.logger import configure
    from optiq_dime import OptiQDIME
    from run_optiq_dime import validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="mujoco_v2_guarded", overrides=["env_name=Humanoid-v4",
            "alg.batch_size=4", "alg.buffer_size=32", "alg.learning_starts=2",
            "alg.actor.learning_starts=2", "diagnostic_interval=1",
            "alg.actor.soft_guard.batch_size=4", "alg.actor.soft_guard.draws=4",
            "alg.actor.soft_guard.components=4", "alg.actor.hidden_dims=[32,32]",
            f"alg.actor.soft_guard.standard_error_multiplier={standard_error_multiplier}",
            "alg.critic.hs=[32,32]"])
    assert validate_config(cfg)
    model = OptiQDIME("MlpPolicy", gym.make("Humanoid-v4", max_episode_steps=2), str(tmp_path), 4, cfg)
    model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
    try:
        model.learn(total_timesteps=8)
        assert model._n_updates == 6 and model.soft_guard_attempts == 6
        assert int(model.policy.actor_state.step) == model.soft_guard_accepts
        assert cfg.alg.behavior_uniform_probability == 0
        with (tmp_path / "logs/progress.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert any(row.get("train/soft_guard_gap_mean") for row in rows)
        assert (tmp_path / "actor_state_8.msgpack").exists()
        # Target actor is updated from the accepted policy only (policy_tau=1).
        for actual, wanted in zip(jax.tree_util.tree_leaves(model.policy.actor_state.params),
                                 jax.tree_util.tree_leaves(model.policy.target_actor_state.params)):
            np.testing.assert_array_equal(actual, wanted)
    finally:
        model.get_env().close()
        model.logger.close()
