# Release checks

This snapshot was checked on an RTX 5090 using Python 3.11, JAX/jaxlib 0.4.33,
Flax 0.9.0, Optax 0.1.7, MuJoCo 2.3.7 and scikit-learn 1.7.2.

- 10 numerical/integration tests passed: truncated-density normalization and
  SciPy agreement, bounded sampling and moments, support stress test, likelihood
  gradient/stop-gradient checks, actor sampling, and short learner updates in
  all five MuJoCo environments. The learner tests fail if OT is invoked.
- 7 configuration tests: five task defaults, explicit ablation overrides, and
  rejection of exploration multipliers other than 0.1.
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
- Metric cleanup checks enforce the retained CSV key set across all five
  environments and verify that MuJoCo evaluation archives contain no success
  or solved-step histories. Detailed diagnostic-only actor calculations were
  removed; candidate weighting and the marginal-likelihood objective remain.
  `env_steps` is the single shared coordinate for all logging backends.
- The learner now uses installed SB3 directly, with no bundled legacy adapter.
  The exploration multiplier is fixed at 0.1 across all five environments;
  alpha adaptation remains enabled and unchanged.

These are correctness/execution checks, not new benchmark results. The tests
used an existing environment; a clean installation of the requirements on
every supported platform has not been validated. Deprecation warnings from
Matplotlib/PyParsing do not affect the recorded test outcomes.

## GMM40 addition

- Combined suite: 20 tests passed (the 17 MuJoCo/configuration tests plus three
  GMM40 target, learner/checkpoint and evaluation-protocol tests).
- The bundled numerical target matches the paper campaign's means/std/weights
  SHA256. Bounded and unbounded reference draws, 10,000 each, match exactly.
- CPU comparison to the archived paper learner: three updates with N=M=64,
  batch=2 and the full 256x3 network gave maximum parameter difference 0.0.
  Full-policy and mean-only outputs (10,000 each) matched within atol=1e-5,
  rtol=1e-6. This is not a full 100K learning-curve reproduction.
- An RTX 5090 CLI smoke test used the actual paper tensor sizes (batch=256,
  N=M=256, 256x3) for two actor updates. It completed with 131,072 training Q
  queries, saved both evaluation views and metrics, generated a PNG, and saved
  checkpoints. Smoke evaluation used 256 samples to bound validation time.
- Checkpoint round-trip preserves the GMM40 optimizer, training key and update
  count; a subsequent update matches uninterrupted execution in the unit test.
