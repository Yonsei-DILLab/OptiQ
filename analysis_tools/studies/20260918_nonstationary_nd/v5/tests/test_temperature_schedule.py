"""Schedule boundaries and the temperature consumed by real v4 updates."""
import copy
import csv
import inspect
import math

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf
from stable_baselines3.common.logger import configure

from optiq_dime.temperature import parse_temperature_schedule, scheduled_temperature
from run_optiq_dime import create_algorithm
from scripts.verify_v4 import verify


@pytest.mark.parametrize("duration", [20000, 40000])
@pytest.mark.parametrize("probability", [0.0, 0.1])
def test_requested_configs_and_boundaries(duration, probability):
    overrides = ['benchmark=ant', f'alg.behavior_uniform_probability={probability}',
                 f'alg.actor.temperature_schedule.anneal_steps={duration}']
    cfg = verify(overrides, 'mujoco_v4_annealing')
    alias = verify(overrides, 'v4/annealing')
    configured = OmegaConf.to_container(cfg, resolve=True)
    expected = OmegaConf.to_container(verify(['benchmark=ant']), resolve=True)
    expected['alg']['actor']['temperature'] = 10.0
    expected['alg']['actor']['temperature_schedule'] = {
        'enabled': True, 'final_temperature': .25, 'anneal_steps': duration}
    expected['alg']['behavior_uniform_probability'] = probability
    for key in ('wandb', 'run_name', 'output_root'):
        expected[key] = configured[key]
    assert configured == expected == OmegaConf.to_container(alias, resolve=True)
    assert cfg.wandb.project == 'v4-test' and cfg.wandb.entity == 'OptiQ'
    schedule = parse_temperature_schedule(cfg.alg.actor, cfg.alg.critic.backup_mode)
    for step in (0, 4999, 5000):
        assert scheduled_temperature(10., schedule, step, 5000) == (10., 0.)
    first, p = scheduled_temperature(10., schedule, 5001, 5000)
    assert .25 < first < 10 and p == 1 / duration
    middle, p = scheduled_temperature(10., schedule, 5000 + duration // 2, 5000)
    assert middle == pytest.approx(math.sqrt(2.5)) and p == .5
    for step in (5000 + duration, 5001 + duration, 1000000):
        assert scheduled_temperature(10., schedule, step, 5000) == (.25, 1.)
    # Calling an earlier step after a later one does not depend on history.
    assert scheduled_temperature(10., schedule, 5000, 5000) == (10., 0.)


@pytest.mark.parametrize("field,value", [
    ('anneal_steps', 0), ('anneal_steps', -1), ('anneal_steps', 1.5),
    ('anneal_steps', True), ('final_temperature', 0),
    ('final_temperature', float('nan')), ('final_temperature', float('inf')),
    ('final_temperature', 11), ('enabled', 'true'),
])
def test_invalid_schedule(field, value):
    actor = {'temperature': 10., 'temperature_schedule': {
        'enabled': True, 'final_temperature': .25, 'anneal_steps': 40000}}
    actor['temperature_schedule'][field] = value
    with pytest.raises(ValueError):
        parse_temperature_schedule(actor, 'td')


def test_disabled_and_soft_backup():
    assert parse_temperature_schedule({'temperature': .1}, 'td') is None
    actor = {'temperature': 10., 'temperature_schedule': {'enabled': False}}
    assert parse_temperature_schedule(actor, 'soft_td') is None
    actor['temperature_schedule'].update(enabled=True, final_temperature=.25, anneal_steps=20)
    with pytest.raises(ValueError, match='plain TD'):
        parse_temperature_schedule(actor, 'soft_td')


@pytest.mark.parametrize("probability", [0.0, 0.1])
def test_real_training_uses_scheduled_temperature_and_preserves_evaluation(tmp_path, monkeypatch, probability):
    cfg = verify(['benchmark=hopper', f'output_root={tmp_path}',
        'alg.batch_size=4', 'alg.buffer_size=32', 'alg.learning_starts=2',
        'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
        'diagnostic_interval=4', 'checkpoint_interval=4',
        'alg.actor.temperature_schedule.anneal_steps=4',
        f'alg.behavior_uniform_probability={probability}'], 'mujoco_v4_annealing')
    saved_config = copy.deepcopy(OmegaConf.to_container(cfg, resolve=True))
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'logs'), ['csv']))
    cb = callbacks.callbacks[0]
    cb.eval_env.envs[0].env._max_episode_steps = 3
    actual, steps = [], []
    original_train = model._train
    signature = inspect.signature(original_train)
    def record_train(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        result = original_train(*args, **kwargs)
        actual.append(float(result[-1]['temperature']))
        steps.append(arguments['n_env_interacts'])
        assert actual[-1] == pytest.approx(arguments['temperature'])
        assert arguments['backup_mode'] == 'td'
        assert float(result[-1]['backup_entropy_term']) == 0.
        return result
    monkeypatch.setattr(model, '_train', record_train)
    try:
        model.learn(total_timesteps=8, callback=callbacks)
        model.logger.dump(8)
        assert steps == list(range(3, 9))
        np.testing.assert_allclose(actual, [*np.geomspace(10., .25, 5)[1:], .25, .25], rtol=1e-6)
        assert model._n_updates == int(model.policy.actor_state.step) == 6
        for leaf in jax.tree_util.tree_leaves(model.policy.actor_state.params):
            assert np.isfinite(np.asarray(leaf)).all()
        assert model.behavior_action_count == (6 if probability else 0)
        assert OmegaConf.to_container(cfg, resolve=True) == saved_config
        assert cb.evaluations_timesteps == [1, 4, 8]
        np.testing.assert_array_equal(cb.histories['zero_z']['env_seeds'],
                                      cb.histories['stochastic_z']['env_seeds'])
        for mode in cb.MODES:
            with np.load(cb.directory/f'evaluations_{mode}.npz') as saved:
                assert saved['results'].shape == (3, 2)
                assert np.isfinite(saved['results']).all()
        with (tmp_path/'logs/progress.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert any(row.get('train/temperature') == '0.25'
                   and row.get('train/temperature_anneal_progress') == '1.0'
                   and row.get('train/temperature_env_steps') == '8' for row in rows)
    finally:
        cb.eval_env.close()
        model.get_env().close()
        model.logger.close()
