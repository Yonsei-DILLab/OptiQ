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
# PointMaze release checks

The separate DrAC PointMaze package passed the following checks on an RTX 5090:

- Full test suite: **25 passed** (including existing MuJoCo/GMM40 tests).
- All three maps: native 4/4/8 goals, horizons 150/300/600, sparse reward,
  training without robustness obstacles and separate obstacle geometry.
- Paper-shaped Simple smoke test: batch4096, N=M64, 256 collectors,
  8192 uniform warmup plus256 transitions, exactly16 learner updates.
  Final actor loss2.075824 and critic loss0.007517 were finite.
- Final code: each of Simple/Medium/Hard completed a reduced6-transition,
  one-update CLI check, normal and obstacle evaluation, and checkpoint output.
- Required vendored simulator, map, utility and XML files compare byte-for-byte
  with the source snapshot. Original third-party license headers are retained.

The paper-shaped check used321f525; the final suite and three-map CLI checks
used3673d59, which adds explicit Python/NumPy seed initialization, corrects task
metadata and restores original trailing whitespace in two upstream files.
These are bounded execution tests, not new1M performance experiments or proof
of bitwise equality with historical training. See `pointmaze/PROTOCOL.md` for
checkpoint/resume limitations.

## PointMaze evaluation parity correction

The earlier release's serial evaluator has been replaced by the original
128-episode-chunk evaluation protocol. Both normal and obstacle evaluations
now run at every checkpoint using the original step-dependent seeds. Five-trial
removal and obstacle robustness, reachable goals, episode lengths and original
raw archive fields are restored. No learner objective or training setting changed.

On RTX5090, the original evaluator/collector excerpt and release evaluator were
run against the same initialized policy on Simple/Medium/Hard, with and without
obstacles, five episodes per condition. All raw trajectory arrays, goal IDs,
returns, lengths and summary values matched exactly; policy RNGs were unchanged.
This validates evaluator equivalence on those inputs, not historical checkpoint
returns or full-training equivalence. Separate tests cover the128+2 episode
boundary, inactive/finished rows, failure inclusion, and exhaustive goal-removal
subsets for four/eight goals. A short CLI run verified both ordinary and obstacle
metrics at an intermediate checkpoint and the final checkpoint. The combined
regression suite passed all27 tests on this implementation (1c6477c).
