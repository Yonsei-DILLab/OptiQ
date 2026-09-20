"""Verify fixed-Q mean initialization with the original fixed learning rate."""
import hashlib
import importlib.util

import flax.serialization
import jax
import numpy as np

from .evaluation import atomic_json
from .optiq import OptiQ
from .target import ROOT, RESULTS, Target


def main():
    assert jax.default_backend() == 'cpu'
    source = RESULTS/'sources/optiq_n16_m64_eps003_i1000_seed0_100k/gmm40/optiq.py'
    spec = importlib.util.spec_from_file_location('frozen_optiq_before_init_lr_cli', source)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    target = Target()
    old = old_module.OptiQ(target, batch=32)
    current = OptiQ(target, batch=32)
    initial_equal = flax.serialization.to_bytes(old.state) == flax.serialization.to_bytes(current.state)
    old.advance(100)
    current.advance(100)
    state_equal = flax.serialization.to_bytes(old.state) == flax.serialization.to_bytes(current.state)
    keys_equal = bool(np.array_equal(old.key, current.key))

    default = OptiQ(target, batch=32)
    mean_init = OptiQ(target, batch=32, mean_output_init_scale=.01)
    checks = dict(default_initial_state_byte_identical=initial_equal,
                  default_100update_state_byte_identical=state_equal,
                  default_100update_key_equal=keys_equal,
                  init_scale_mu_kernel_tenfold=bool(np.allclose(mean_init.state.params['mu']['kernel'],
                                                               10*default.state.params['mu']['kernel'], atol=1e-8)),
                  init_scale_sigma_head_unchanged=flax.serialization.to_bytes(mean_init.state.params['log_std'])
                                                == flax.serialization.to_bytes(default.state.params['log_std']))
    result = dict(passed=all(checks.values()), checks=checks, backend=jax.default_backend(),
                  old_source=str(source), old_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  new_source_sha256=hashlib.sha256((ROOT/'gmm40/optiq.py').read_bytes()).hexdigest(),
                  scope='CPU B32 N16/M64 100-update default regression and controlled mean initialization. Learning rate fixed at 3e-4. No saved run overwritten.')
    atomic_json(RESULTS/'diagnostics/optiq_mean_init_default_validation_20260918.json', result)
    print(result, flush=True)
    assert result['passed'], result


if __name__ == '__main__':
    main()
