"""Analytic checks for the auxiliary evaluator, independent of OptiQ training."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdiffeq")

from benchmarks.gmm40.idem_evaluate import sample_cfm


class LinearVelocity(torch.nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, t, x):
        return self.coefficient * x


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
@pytest.mark.parametrize("batch_size", [None, 4])
def test_cfm_sampling_matches_analytic_flow_and_preserves_rng(coefficient, batch_size):
    torch.manual_seed(123)
    prior = torch.randn(17, 2) * 50.0
    before = torch.random.get_rng_state().clone()
    actual = sample_cfm(
        LinearVelocity(coefficient), 17, 50.0, "cpu", 1e-7, 123, batch_size
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    expected = prior * math.exp(coefficient)
    # Even zero velocity incurs float32 dense-interpolation rounding in odeint.
    # Check relative as well as absolute error at the original-coordinate scale.
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
