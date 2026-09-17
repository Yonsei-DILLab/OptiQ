"""Requested v5 exploration profiles: only collection/schedule differ."""
import copy
import inspect

import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from run_optiq_dime import create_algorithm
from scripts.verify_v5 import verify

PROFILES = [('behavior010', .1, False), ('annealing', 0., True),
            ('annealing_behavior010', .1, True)]


@pytest.mark.parametrize('profile,probability,annealing', PROFILES)
@pytest.mark.parametrize('seed', range(4))
def test_exact_profile_difference_and_alias(profile, probability, annealing, seed):
    overrides = ['benchmark=ant', f'seed={seed}']
    cfg = OmegaConf.to_container(verify(overrides, 'mujoco_v5_'+profile), resolve=True)
    alias = OmegaConf.to_container(verify(overrides, 'v5/'+profile), resolve=True)
    expected = OmegaConf.to_container(verify(overrides), resolve=True)
    expected['alg']['behavior_uniform_probability'] = probability
    if annealing:
        expected['alg']['actor']['temperature'] = 10.
        expected['alg']['actor']['temperature_schedule'] = {
            'enabled':True, 'final_temperature':.25, 'anneal_steps':40000}
    for key in ('wandb','output_root','run_name'):
        expected[key] = cfg[key]
    assert cfg == expected == alias
    assert cfg['alg']['actor']['ot_student_action'] == 'mean'
    assert cfg['alg']['actor']['sinkhorn_epsilon'] == .1
    assert cfg['alg']['optimizer']['ac_grad_norm'] is None
    assert cfg['wandb']['project'] == 'v4-test'


@pytest.mark.parametrize('profile,probability,annealing', PROFILES)
def test_ant_update_temperature_mean_ot_and_collection_isolation(tmp_path, monkeypatch, profile, probability, annealing):
    overrides=['benchmark=ant', f'output_root={tmp_path}', 'alg.batch_size=4',
        'alg.buffer_size=32', 'alg.learning_starts=2', 'alg.actor.learning_starts=2',
        'num_eval_episodes=2', 'eval_interval=4', 'diagnostic_interval=4', 'checkpoint_interval=4']
    if annealing:
        overrides += ['alg.actor.temperature_schedule.anneal_steps=4']
    cfg = verify(overrides,'mujoco_v5_'+profile)
    before_config=copy.deepcopy(OmegaConf.to_container(cfg,resolve=True))
    model,callbacks=create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'test_logs'), ['csv']))
    cb=callbacks.callbacks[0]
    cb.eval_env.envs[0].env._max_episode_steps=2
    model.get_env().envs[0].env._max_episode_steps=2
    temperatures=[]
    original=model._train
    signature=inspect.signature(original)
    def record_train(*args,**kwargs):
        bound=signature.bind(*args,**kwargs).arguments
        assert bound['ot_student_action']=='mean' and bound['backup_mode']=='td'
        result=original(*args,**kwargs)
        assert float(result[-1]['backup_entropy_term'])==0
        temperatures.append(float(result[-1]['temperature']))
        return result
    monkeypatch.setattr(model,'_train',record_train)
    actions=[]
    env=model.get_env().envs[0]
    step=env.step
    def record_step(action):
        actions.append(np.array(action,copy=True));return step(action)
    monkeypatch.setattr(env,'step',record_step)
    try:
        model.learn(total_timesteps=8,callback=callbacks)
        expected=[*np.geomspace(10.,.25,5)[1:],.25,.25] if annealing else [.25]*6
        np.testing.assert_allclose(temperatures,expected,rtol=1e-6)
        assert model._n_updates==int(model.policy.actor_state.step)==6
        assert model.behavior_action_count==(6 if probability else 0)
        np.testing.assert_allclose(model.policy.unscale_action(model.replay_buffer.actions[:8,0]),actions,atol=1e-6)
        for mode in cb.MODES:
            assert np.asarray(cb.histories[mode]['results'][-1]).shape == (2,)
        np.testing.assert_array_equal(cb.histories['zero_z']['env_seeds'],cb.histories['stochastic_z']['env_seeds'])
        rng=copy.deepcopy(model.behavior_rng.bit_generator.state)
        def fail(*args,**kwargs):
            raise AssertionError('Evaluation must bypass uniform collection')
        monkeypatch.setattr(model,'_sample_action',fail)
        cb.n_calls=12;cb.num_timesteps=12;cb._on_step()
        assert model.behavior_rng.bit_generator.state==rng
        assert OmegaConf.to_container(cfg,resolve=True)==before_config
    finally:
        cb.eval_env.close();model.get_env().close();model.logger.close()
