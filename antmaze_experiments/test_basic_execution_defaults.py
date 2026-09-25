"""Check the selected AntMaze OptiQ defaults without launching training."""
import unittest
from types import SimpleNamespace

from .collection_profile import get_profile
from .learners import BATCH
from .run import select_execution_defaults
from .settings import WARMUP


class BasicExecutionDefaultsTest(unittest.TestCase):
    def arguments(self, method='optiq', profile='basic', **overrides):
        values = dict(method=method, optiq_config_profile=profile,
                      temperature=None, dacer=None, noveld=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_basic_optiq_and_vectorized_data_profile(self):
        settings = select_execution_defaults(self.arguments())
        self.assertEqual((settings.temperature, settings.dacer, settings.noveld),
                         (1.0, 'off', 'off'))
        self.assertEqual(get_profile(),
                         dict(num_envs=256, updates_per_vector_step=8))
        self.assertEqual((BATCH, WARMUP), (4096, 8192))
        self.assertEqual(8 / 256, 1 / 32)
        self.assertEqual(8 * BATCH / 256, 128)

    def test_explicit_experiment_overrides_are_preserved(self):
        settings = select_execution_defaults(self.arguments(
            temperature=3.0, dacer='on', noveld='on'))
        self.assertEqual((settings.temperature, settings.dacer, settings.noveld),
                         (3.0, 'on', 'on'))

    def test_legacy_and_native_baseline_defaults_remain_explicit(self):
        legacy = select_execution_defaults(self.arguments(profile='legacy'))
        self.assertEqual((legacy.temperature, legacy.dacer, legacy.noveld),
                         (None, 'on', 'on'))
        sac = select_execution_defaults(self.arguments(method='sac'))
        self.assertIsNone(sac.temperature)
        self.assertEqual(sac.noveld, 'on')


if __name__ == '__main__':
    unittest.main()
