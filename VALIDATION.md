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
- Original-versus-release regression on CPU, with matched seeds and replay
  sampling, produced exactly equal actor and critic parameters after four
  updates in Ant and Humanoid (batch 4, two warmup transitions).
- GPU parameter parity at rtol=1e-5 and atol=1e-6 did not pass. In the Ant
  four-update check, initial parameters matched exactly, but the largest final
  parameter difference was 4.68e-4. Refactoring changes the compiled computation
  graph; CPU parity does not establish bitwise GPU equivalence or equal long-run
  returns. No new performance reproduction claim is made.
- A short Hopper CLI run with exploration enabled completed training, both
  evaluation modes, diagnostics and checkpoint output.

These are correctness/execution checks, not new benchmark results. The tests
used an existing environment; a clean installation of the requirements on
every supported platform has not been validated. Deprecation warnings from
Matplotlib/PyParsing do not affect the recorded test outcomes.
