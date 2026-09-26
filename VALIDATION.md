# Release checks

This snapshot was checked on an RTX 5090 using Python 3.11, JAX/jaxlib 0.4.33,
Flax 0.9.0, Optax 0.1.7, MuJoCo 2.3.7 and scikit-learn 1.7.2.

- 10 numerical/integration tests passed: truncated-density normalization and
  SciPy agreement, bounded sampling and moments, support stress test, likelihood
  gradient/stop-gradient checks, actor sampling, and short learner updates in
  all five MuJoCo environments. The learner tests fail if OT is invoked.
- 6 configuration tests passed: five task defaults and explicit ablation overrides.
- The release is restricted to the trainable-scale direct mixture-likelihood
  experiment. Historical branches and inactive configuration keys were removed.
  Numerical regression against the original learner is used to check the
  refactored actor and scalar-critic update path.

These are correctness/execution checks, not new benchmark results. The tests
used an existing environment; a clean installation of the requirements on
every supported platform has not been validated. Deprecation warnings from
Matplotlib/PyParsing do not affect the recorded test outcomes.
