"""Run directly to avoid the training suite's online W&B fixture."""
import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

spec = importlib.util.spec_from_file_location(
    'optiq_optimizers', Path(__file__).resolve().parents[1] / 'optiq_dime/optimizers.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GradientClippingTests(unittest.TestCase):
    def test_global_across_leaves_and_twins_before_adam(self):
        grads = {'q1': jnp.array([3.]), 'q2': jnp.array([4.])}
        params = jax.tree_util.tree_map(jnp.zeros_like, grads)
        for limit in (2., 10.):
            tx = module.adam_with_grad_clip(.0003, .9, .999, limit)
            _, state = jax.jit(tx.update)(grads, tx.init(params), params)
            expected = jax.tree_util.tree_map(lambda g: g * min(1., limit / 5.), grads)
            # Adam's first moment stores the clipped gradient, not the raw one.
            for actual, want in zip(jax.tree_util.tree_leaves(state[1][0].mu),
                                    jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(actual, .1 * want, rtol=1e-6)
            self.assertLessEqual(float(optax.global_norm(expected)), limit + 1e-6)

    def test_disabled_is_original_adam(self):
        params = {'x': jnp.zeros(2)}; grads = {'x': jnp.array([3., 4.])}
        a = module.adam_with_grad_clip(.0003, .9, .999, None)
        b = optax.adam(.0003, b1=.9, b2=.999)
        for x, y in zip(jax.tree_util.tree_leaves(a.update(grads, a.init(params), params)),
                        jax.tree_util.tree_leaves(b.update(grads, b.init(params), params))):
            np.testing.assert_array_equal(x, y)

    def test_bad_limits(self):
        for limit in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                module.adam_with_grad_clip(.0003, .9, .999, limit)


if __name__ == '__main__':
    unittest.main()
