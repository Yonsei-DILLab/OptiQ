from types import SimpleNamespace
import numpy as np

from scripts.evaluate_v2_confirmation import evaluate_episodes, paired_summary, pool_environment_metrics


def test_time_limit_episodes_are_included_in_checkpoint_evaluation():
    class Model:
        policy = SimpleNamespace(key=None)
        def predict(self, obs, deterministic):
            assert deterministic is False
            return np.zeros(1), None
    class Environment:
        def reset(self, seed):
            self.timeout = seed % 2 == 1
            return np.zeros(1), {}
        def step(self, action):
            return np.zeros(1), 100. if self.timeout else 1., not self.timeout, self.timeout, {}
    episodes = evaluate_episodes(Model(), Environment(), 2, 970000)
    assert [e['return'] for e in episodes] == [1., 100.]
    assert episodes[1]['truncated'] and not episodes[1]['terminated']
    assert paired_summary([{'training_seed': 0, 'method': 'v2', 'episodes': episodes}], 970000) == {'complete': False}


def test_reward_components_include_terminal_steps_and_pool_by_step():
    class Model:
        policy = SimpleNamespace(key=None)
        def predict(self, obs, deterministic):
            return np.zeros(1), None
    class Environment:
        def reset(self, seed):
            self.horizon = 3 if seed % 2 else 1
            self.elapsed = 0
            return np.zeros(1), {}
        def step(self, action):
            self.elapsed += 1
            velocity = 1. if self.horizon == 1 else 3.
            info = {'reward_alive': 5., 'reward_linvel': velocity,
                    'reward_quadctrl': -1., 'x_velocity': velocity}
            return np.zeros(1), 4.+velocity, False, self.elapsed == self.horizon, info
    episodes = evaluate_episodes(Model(), Environment(), 2, 970000)
    assert [e['length'] for e in episodes] == [1, 3]
    assert [e['return'] for e in episodes] == [5., 21.]
    for e in episodes:
        components = e['environment_metrics']
        assert sum(components[k]['sum'] for k in ['reward_alive', 'reward_linvel', 'reward_quadctrl']) == e['return']
    metrics = pool_environment_metrics(episodes)
    assert metrics['x_velocity'] == {'sum': 10., 'count': 4, 'mean': 2.5}
    assert metrics['reward_alive']['sum'] == 20.
    assert pool_environment_metrics([{'environment_metrics': {}}]) == {}
