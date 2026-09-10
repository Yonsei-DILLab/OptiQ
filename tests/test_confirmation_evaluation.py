from types import SimpleNamespace
import numpy as np

from scripts.evaluate_v2_confirmation import evaluate_episodes, paired_summary


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
