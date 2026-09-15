"""Behavior mixture endpoints, per-environment routing, and real update parity."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.behavior import parse_behavior_best_of_k_probability
from run_optiq_dime import create_algorithm, validate_config
from scripts.verify_v5 import verify


class FixedRng:
    def __init__(self, draws):
        self.draws = np.asarray(draws)
        self.calls = 0

    def random(self, count):
        self.calls += 1
        assert count == len(self.draws)
        return self.draws


def collector(probability=.5, timestep=5000, start=0, draws=(.1, .9, .2)):
    model = object.__new__(OptiQDIME)
    model.behavior_uniform_probability = 0.
    model.behavior_best_of_k = 8
    model.behavior_best_of_k_start_step = start
    model.behavior_best_of_k_probability = probability
    model.behavior_best_of_k_rng = FixedRng(draws)
    model.behavior_best_of_k_key = jax.random.PRNGKey(7)
    model.behavior_best_of_k_count = model.behavior_best_of_k_opportunity_count = 0
    model.num_timesteps = timestep
    model._last_obs = np.arange(9, dtype=np.float32).reshape(3, 3)
    model.policy = SimpleNamespace(actor_state=None, qf_state=None,
        prepare_obs=lambda obs: (obs, False), unscale_action=lambda a: 2+3*a)
    logged = {}
    model.set_logger(SimpleNamespace(record=lambda k, v: logged.__setitem__(k, v)))
    return model, logged


def test_mask_routes_only_selected_states_and_preserves_original_gaussian_actions():
    model, logged = collector()
    base = np.array([[.1, .2], [.3, .4], [.5, .6]], dtype=np.float32)
    physical = model.policy.unscale_action(base)
    key = model.behavior_best_of_k_key.copy()
    with patch.object(OffPolicyAlgorithm, '_sample_action', return_value=(physical, base)), \
         patch('optiq_dime.algorithm.select_best_of_k',
             side_effect=lambda a, c, obs, first, key, k: (-first, {'best_of_k_q_gain': 1.})) as selector:
        action, stored = model._sample_action(5000, n_envs=3)
    expected = base.copy()
    expected[[0, 2]] *= -1
    np.testing.assert_array_equal(stored, expected)
    np.testing.assert_array_equal(action, 2+3*expected)
    np.testing.assert_array_equal(selector.call_args.args[2], model._last_obs[[0, 2]])
    np.testing.assert_array_equal(selector.call_args.args[3], base[[0, 2]])
    np.testing.assert_array_equal(model.behavior_best_of_k_key, jax.random.split(key)[0])
    assert model.behavior_best_of_k_count == 2 and model.behavior_best_of_k_opportunity_count == 3
    assert logged['rollout/best_of_k_fraction'] == pytest.approx(2/3)
    assert logged['rollout/behavior_best_of_k_probability'] == .5


def test_mixture_rejects_additional_noise_before_random_branch_decision():
    model, _ = collector(draws=(.99, .99, .99))
    with patch.object(OffPolicyAlgorithm, '_sample_action', return_value=(np.ones((3,2)), np.zeros((3,2)))):
        with pytest.raises(ValueError, match='additional action_noise'):
            model._sample_action(5000, action_noise=object(), n_envs=3)
    assert model.behavior_best_of_k_rng.calls == 0


@pytest.mark.parametrize('probability,timestep,start,expected_rng_calls', [
    (0., 5000, 0, 0), (.5, 4999, 0, 0), (.5, 5000, 6000, 0), (.5, 5000, 0, 1),
])
def test_unselected_or_inactive_collection_keeps_original_objects_and_selection_rng(
        probability, timestep, start, expected_rng_calls):
    model, _ = collector(probability, timestep, start, (.99, .99, .99))
    actions, replay = np.ones((3, 2)), np.zeros((3, 2))
    key = model.behavior_best_of_k_key.copy()
    with patch.object(OffPolicyAlgorithm, '_sample_action', return_value=(actions, replay)), \
         patch('optiq_dime.algorithm.select_best_of_k', side_effect=AssertionError('unexpected selector')):
        actual, stored = model._sample_action(5000, n_envs=3)
    assert actual is actions and stored is replay
    np.testing.assert_array_equal(key, model.behavior_best_of_k_key)
    assert model.behavior_best_of_k_count == 0
    assert model.behavior_best_of_k_rng.calls == expected_rng_calls


@pytest.mark.parametrize('value', [-.1, 1.1, True, None, 'invalid', float('nan'), float('inf')])
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValueError, match='behavior_best_of_k_probability'):
        parse_behavior_best_of_k_probability({'behavior_best_of_k':8,
            'behavior_best_of_k_probability':value})


def test_profile_preserves_all_original_optiq_components_and_labels_mixture():
    original = OmegaConf.to_container(verify([], 'mujoco_v5_bestk_combined'), resolve=True)
    mixed = OmegaConf.to_container(verify([], 'mujoco_v5_bestk_mixed'), resolve=True)
    expected = copy.deepcopy(original)
    expected['alg']['behavior_best_of_k_probability'] = .5
    for key in ('run_name', 'output_root', 'wandb'):
        expected[key] = mixed[key]
    assert mixed == expected == OmegaConf.to_container(verify([], 'v5/bestk_mixed'), resolve=True)
    with pytest.raises(ValueError, match='collection probability matches profile'):
        verify(['+alg.behavior_best_of_k_probability=.5'], 'mujoco_v5_bestk_combined')
    with pytest.raises(ValueError, match='requires best-of-k'):
        parse_behavior_best_of_k_probability({'behavior_best_of_k':1, 'behavior_best_of_k_probability':.5})
    with pytest.raises(ValueError, match='uniform replacement'):
        verify(['alg.behavior_uniform_probability=.1'], 'mujoco_v5_bestk_mixed')


@pytest.mark.parametrize('task', ['ant', 'hopper'])
@pytest.mark.parametrize('probability', [0., 1.])
def test_endpoints_preserve_actual_training_parameters_replay_rng_and_evaluation(tmp_path, task, probability):
    snapshots = []
    base_profile = 'mujoco_v5_bestk_proposal' if probability == 0 else 'mujoco_v5_bestk_combined'
    for variant in range(2):
        output = tmp_path/str(variant)
        cfg = verify([f'benchmark={task}', 'seed=3', f'output_root={output}',
            'alg.batch_size=4', 'alg.buffer_size=32', 'alg.learning_starts=2',
            'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
            'diagnostic_interval=4', 'checkpoint_interval=4',
            'alg.actor.temperature=.01' if task == 'hopper' else 'alg.actor.temperature=.25'],
            base_profile if variant == 0 else 'mujoco_v5_bestk_combined')
        if variant == 1:
            OmegaConf.update(cfg, 'alg.behavior_best_of_k_probability', probability, force_add=True)
        validate_config(cfg)
        model, callbacks = create_algorithm(cfg)
        model.set_logger(configure(str(output/'logs'), ['csv']))
        callback = callbacks.callbacks[0]
        callback.eval_env.envs[0].env._max_episode_steps = 2
        model.get_env().envs[0].env._max_episode_steps = 2
        mixture_state = copy.deepcopy(model.behavior_best_of_k_rng.bit_generator.state)
        try:
            model.learn(total_timesteps=8, callback=callbacks)
            assert model.behavior_best_of_k_rng.bit_generator.state == mixture_state
            snapshots.append(dict(
                observations=model.replay_buffer.observations[:8].copy(),
                actions=model.replay_buffer.actions[:8].copy(),
                rewards=model.replay_buffer.rewards[:8].copy(),
                actor=jax.tree.leaves((model.policy.actor_state.params, model.policy.actor_state.opt_state,
                    model.policy.target_actor_state.params)),
                critic=jax.tree.leaves((model.policy.qf_state.params, model.policy.qf_state.opt_state,
                    model.policy.qf_state.target_params)),
                policy_rng=model.policy.key.copy(), noise_rng=model.policy.noise_key.copy(),
                training_rng=model.key.copy(), selection_rng=model.behavior_best_of_k_key.copy(),
                replay_rng=copy.deepcopy(np.random.get_state()), evaluation=copy.deepcopy(callback.histories)))
        finally:
            callback.eval_env.close()
            model.get_env().close()
            model.logger.close()
    a, b = snapshots
    for key in ('observations', 'actions', 'rewards', 'policy_rng', 'noise_rng', 'training_rng', 'selection_rng'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('actor', 'critic', 'replay_rng'):
        assert len(a[key]) == len(b[key])
        for x, y in zip(a[key], b[key]):
            np.testing.assert_array_equal(x, y)
    assert a['evaluation'] == b['evaluation']
