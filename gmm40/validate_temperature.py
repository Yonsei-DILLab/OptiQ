"""Validate temperature-only fixed-Q control without changing the v5 core."""
import hashlib
import importlib.util

import flax.serialization
import jax
import numpy as np

from .evaluation import atomic_json
from .optiq import OptiQ
from .target import ROOT, RESULTS, Target


def max_difference(a,b):
    return max(float(np.abs(np.asarray(x)-np.asarray(y)).max())
               for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def main():
    assert jax.default_backend()=='cpu'
    source=RESULTS/'sources/optiq_n16_m64_meaninit001_seed0_5k/gmm40/optiq.py'
    spec=importlib.util.spec_from_file_location('frozen_optiq_before_temperature_control',source)
    old_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    target=Target()
    old=old_module.OptiQ(target,batch=32)
    current=OptiQ(target,batch=32)
    initial_equal=flax.serialization.to_bytes(old.state)==flax.serialization.to_bytes(current.state)
    old.advance(100);current.advance(100)
    checks=dict(default_initial_state_byte_identical=initial_equal,
                default_100update_state_byte_identical=flax.serialization.to_bytes(old.state)==flax.serialization.to_bytes(current.state),
                default_100update_key_equal=bool(np.array_equal(old.key,current.key)))
    # Independent unchanged adapter receives scaled oracle Q. Its proposal density
    # is unscaled, so this also checks that temperature affects only Q, not log q.
    class ScaledOracle:
        def jax_log_prob(self,x):
            return target.jax_log_prob(x)/.01
    cold=OptiQ(target,batch=32,temperature=.01)
    reference=old_module.OptiQ(ScaledOracle(),batch=32)
    cold_info=cold.advance(100);reference.advance(100)
    parameter_error=max_difference(cold.state.params,reference.state.params)
    optimizer_error=max_difference(cold.state.opt_state,reference.state.opt_state)
    checks.update(T001_scaled_oracle_parameter_match=parameter_error<2e-5,
                  T001_scaled_oracle_optimizer_match=optimizer_error<2e-5,
                  T001_reference_rng_equal=bool(np.array_equal(cold.key,reference.key)),
                  T001_all_metrics_finite=all(np.isfinite(v) for v in cold_info.values()),
                  T001_changes_training=max_difference(cold.state.params,current.state.params)>1e-5)
    result=dict(passed=all(checks.values()),checks=checks,backend=jax.default_backend(),
                parameter_max_difference=parameter_error,optimizer_max_difference=optimizer_error,
                T001_100update_metrics=cold_info,
                old_source=str(source),old_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                new_source_sha256=hashlib.sha256((ROOT/'gmm40/optiq.py').read_bytes()).hexdigest(),
                scope='CPU B32 N16/M64: default 100-update regression and T=.01 vs unchanged adapter with Q/.01 oracle. No saved checkpoint modified; learning rate remains 3e-4.')
    atomic_json(RESULTS/'diagnostics/optiq_temperature_validation_20260918.json',result)
    print(result,flush=True)
    assert result['passed'],result


if __name__=='__main__':main()
