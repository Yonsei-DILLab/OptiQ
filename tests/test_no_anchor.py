import json
from pathlib import Path
import subprocess
import types

import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, KVWriter

from optiq_dime import OptiQDIME
from optiq_dime.evaluation import MujocoEvalCallback
from optiq_dime.transport import TruncatedGaussianKDE
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="optiq_dime_no_anchor", overrides=list(overrides))


@pytest.mark.parametrize("benchmark", ["pen_twirl_hard", "ant", "humanoid"])
def test_supplied_no_anchor_reference(benchmark):
    reference = json.loads((ROOT / "tests/data/no_anchor_reference.json").read_text())
    cfg = config([f"benchmark={benchmark}"])
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)

    def compare(expected, current, prefix=""):
        for key, value in expected.items():
            path = f"{prefix}.{key}" if prefix else key
            if path == "alg.actor.include_anchor":
                assert value is True and current[key] is False  # prose takes precedence
            elif path == "alg.actor.proposals_per_policy_sample":
                assert value == 5 and current[key] == 4  # confirmed 16 x 64 OT
            elif isinstance(value, dict):
                compare(value, current[key], path)
            else:
                assert value == current[key], path

    compare({"alg": reference["alg"]}, actual)
    compare(reference["evaluation"], actual)
    for key, value in reference["runtime"].items():
        if benchmark == "pen_twirl_hard" or key in {"use_jit", "total_steps"}:
            assert actual[key] == value
    assert cfg.alg.actor.proposal_sampling_mode == "stratified"
    assert cfg.alg.actor.density_beta == 0.1
    assert cfg.alg.actor.adaptive_density_beta is False
    assert cfg.alg.actor.num_policy_samples * cfg.alg.actor.proposals_per_policy_sample == 64
    assert cfg.seed == 0


def test_default_queue_uses_seeds_zero_one_two():
    from scripts.no_anchor_runs import tasks, command, BENCHMARKS, DEFAULT_SEEDS
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert [(t.benchmark, t.seed) for t in tasks()] == [("pen_twirl_hard", s) for s in (0, 1, 2)]
    table = tasks(BENCHMARKS)
    assert len(table) == len(set(table)) == 9
    for task in table:
        cfg = config(command(task)[3:])
        assert validate_config(cfg)
        assert cfg.seed == task.seed
        assert cfg.alg.actor.include_anchor is False


def test_parameter_alias_and_unsupported_loss_rejected():
    cfg = config(["alg.actor.density_correction_beta=0.25"])
    assert cfg.alg.actor.density_beta == 0.25
    assert validate_config(cfg)
    with pytest.raises(ValueError, match="must agree"):
        validate_config(config(["alg.actor.density_beta=0.5"]))
    with pytest.raises(ValueError, match="pointwise_mse"):
        validate_config(config(["alg.actor.distillation_loss=ignored_loss"]))


@pytest.mark.parametrize("anchor", [False, True])
def test_kde_sampling_and_density_match_original_distribution(anchor):
    original = types.ModuleType("reference_transport")
    source = subprocess.check_output(
        ["git", "show", "c6132fd:optiq_dime/transport.py"], cwd=ROOT, text=True)
    exec(compile(source, "reference_transport.py", "exec"), original.__dict__)
    centers = jax.random.uniform(jax.random.PRNGKey(2), (32, 16, 3), minval=-1, maxval=1)
    kde = TruncatedGaussianKDE.from_centers(centers, 0.2, 0.5)
    samples = kde.sample_stratified(jax.random.PRNGKey(3), 5, include_anchor=anchor)
    expected = original.sample_truncated_gaussian(jax.random.PRNGKey(3), centers, 5, 0.2, 0.5, anchor)
    np.testing.assert_array_equal(samples, expected)
    assert samples.shape == (32, 16, 5, 3)
    assert np.max(np.abs(np.asarray(samples) - np.asarray(centers)[:, :, None])) <= 0.500001
    assert np.max(np.abs(samples)) <= 1.000001
    if not anchor:
        assert not np.any(np.all(np.asarray(samples) == np.asarray(centers)[:, :, None], axis=-1))
    flat = samples.reshape(32, 80, 3)
    np.testing.assert_allclose(kde.log_prob(flat), original.truncated_mixture_log_density(
        flat, centers, 0.2, 0.5), rtol=1e-6, atol=1e-6)


def tiny_model(overrides=()):
    cfg = config(["benchmark=ant", "alg.critic.hs=[32,32]", "alg.actor.hidden_dims=[32,32]",
                  "alg.buffer_size=32", "alg.batch_size=4", *overrides])
    model = OptiQDIME("MlpPolicy", gym.make("Ant-v4"), None, 1, cfg)
    model.set_logger(configure(None, []))
    return model


def test_actor_warmup_and_diagnostic_cadence():
    model = tiny_model(["alg.learning_starts=2", "alg.actor.learning_starts=6", "diagnostic_interval=4"])
    class Capture(KVWriter):
        def __init__(self):
            self.records = []
        def write(self, values, excluded, step=0):
            self.records.append((step, dict(values)))
        def close(self):
            pass
    capture = Capture()
    model.logger.output_formats.append(capture)
    before = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    critic_before = jax.tree_util.tree_leaves(model.policy.qf_state.params)
    try:
        model.learn(total_timesteps=6)
        for x, y in zip(before, jax.tree_util.tree_leaves(model.policy.actor_state.params)):
            np.testing.assert_array_equal(x, y)
        assert any(not np.array_equal(x, y) for x, y in zip(
            critic_before, jax.tree_util.tree_leaves(model.policy.qf_state.params)))
        model.learn(total_timesteps=2, reset_num_timesteps=False)
        assert any(not np.array_equal(x, y) for x, y in zip(
            before, jax.tree_util.tree_leaves(model.policy.actor_state.params)))
        diagnostics = [(step, values) for step, values in capture.records if 'train/source_ess_absolute' in values]
        assert [step for step, _ in diagnostics] == [4, 8]
        assert diagnostics[-1][1]['train/density_beta_mean'] == pytest.approx(0.1)
        assert 1 <= diagnostics[-1][1]['train/source_ess_absolute'] <= 64
        assert all('train/local_anchor_argmax_fraction' not in values for _, values in capture.records)
    finally:
        model.get_env().close()


@pytest.mark.parametrize("solved_count,success", [(0, False), (5, False), (6, True)])
def test_myo_success_counts_full_episode_and_persists(tmp_path, solved_count, success):
    class SolvedEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (27,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.step_index = 0
            return np.zeros(27, np.float32), {}
        def step(self, action):
            solved = self.step_index % 2 == 0 and self.step_index // 2 < solved_count
            self.step_index += 1
            return np.zeros(27, np.float32), 1.0, self.step_index == 12, False, {'solved': solved}
    model = tiny_model()
    env = make_vec_env(SolvedEnv)
    callback = MujocoEvalCallback(env, config(["num_eval_episodes=1"]), tmp_path)
    try:
        callback.init_callback(model)
        callback.n_calls = callback.num_timesteps = 1
        callback._on_step()
        data = np.load(tmp_path / 'evaluations.npz')
        assert data['successes'].tolist() == [[success]]
        assert data['solved_steps'].tolist() == [[solved_count]]
        assert data['ep_lengths'].tolist() == [[12]]
    finally:
        env.close()
        model.get_env().close()
