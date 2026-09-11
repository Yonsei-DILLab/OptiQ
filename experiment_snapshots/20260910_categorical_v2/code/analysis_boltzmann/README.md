# Boltzmann backup analysis

This package adds experiments without modifying production OptiQ. Base commit:
`7e2da67d2f0988f6f211b635f7311d32a0c8e8c6` (`mujoco-setting`).

## Experiments

- `frozen`: calls **production `OptiQDIME.update_actor`** against a fixed analytic
  scalar Q exposed as two identical critics. Temperature .25, density correction
  1, 16x4 candidates, clipped implicit actor, Sinkhorn .05/30, row-argmax MSE.
  Default initialization and a separately labelled reference-informed coverage
  warm start (8192 quantile-map pairs, 2000 supervised updates) are both tested.
  Coverage warm starts are an experimental aid, not a capability claim.
- `movecar`: production OptiQ actor/critic training; matched 256x3 GELU DDPG,
  SD2, TD3, SD3 baselines. Deterministic baselines keep tanh outputs, .005 target
  updates and .5 exploration noise. OptiQ retains its existing clipped output,
  policy noise, hard target-actor copy, live mean extraction / target min backup,
  and truncated TD noise. SD2/SD3 use K=50 and inverse temperature 4. Main local
  backup is no-IS; `--importance official_is` selects pre-clipping Gaussian IS.
  Baselines are JAX adaptations, not bitwise reproductions of the original code.
  SD3 updates both branches synchronously; TD3 uses policy delay 2; SD3 delay 1.
- `landscape`: **value objective, not the OT regression loss**. OptiQ uses
  `-E[mean(Q1,Q2)(s,g(s,z))]`; deterministic first-actor objectives use Q1.
  A common fixed 256-state grid and 32 latent draws/state are used, with 256
  random unit directions, radii .01/.05/.1 times parameter norm, and 101 path
  points. Early/final checkpoints have the same architecture and seed. A
  cross-critic path is supported with `--critics method:path ...`.
- `control`: common fixed 65536-transition replay, same initial scalar critic and
  implicit actor, max/grid Boltzmann/actor/local backup only. Actor extracts the
  same target Q used in backup evaluation, no TD noise. Single Q is duplicated
  only at the interface to production OT. The local arm is given the **global
  grid-max center**, favoring the local estimator. This is not default OptiQ.
- `learned_backup`: all methods evaluate the **same frozen target twin-min Q**
  from an OptiQ checkpoint. The production actor's live-mean/target-min mismatch
  remains, explicitly separated from the analytic-Q actor-fitting experiment.
- `plots`: array-only regeneration, PDF + PNG. No retraining.

## Numerical definitions

Action domain is [-1,1]^d, measure is Lebesgue, tau=.25.
`B(Q) = integral exp(Q/tau) Q / integral exp(Q/tau)`.
Frozen-Q truth uses midpoint quadrature doubled until successive backup values
change by <1e-4 (otherwise fail). High-dimensional Q is a sum of independent 1D
energies, so truth factorizes exactly; Q range grows with dimension. Do not
interpret the dimension sweep as a fixed-total-Q-range experiment.

Local estimates:
1. no-IS: softmax over Q values of clipped Gaussian candidates;
2. official-IS: Gaussian density of **unclipped noise**, as in public SD3;
3. truncated-IS: actual normalized truncated Gaussian proposal density.
The third converges to a **local-support** Boltzmann expectation, not the global
one. Finite self-normalized IS is not claimed to be unbiased.

K = 1,8,16,50,64,256,1024; 2000 independent repeated estimates per condition.
Per-fit repetition variance is distinct from across-training-seed confidence
intervals. Modes have nearest standardized-center labels, consistently used
for target and actor. Separable cases use Cartesian labels. Distribution files
preserve mode masses even when a backup mean happens to be accurate.

## MoveCar and bias interpretation

x'=clip(x+a,0,10), reward at x', regions [0.5,1.5] -> 2 and [8.5,9.5] -> 1.
Reset x=8; 100-step time limit is truncation, not terminal in TD learning.
Training 1M steps, uniform warmup 5000, batch256, lr3e-4, gamma .99, UTD1.
Task return is undiscounted 100-step return. Value probes use exact first action
and 2048-step continuation, with maximum discounted tail <3e-7.

The value/bias curve is `estimated Q - return of the frozen action-selection
policy without exploration noise`. For SD and default OptiQ, this can include a
backup-policy/behavior-policy mismatch. It is **not** labelled pure critic
approximation bias or Boltzmann integration error. The separate common-Q backup
experiment measures integration/actor errors against quadrature directly.

## Reproducibility and execution

Each run writes the resolved OptiQ reference config, effective CLI arguments, relevant source SHA256,
package versions, independent train/evaluation seeds, checkpoints, raw CSV/NPZ,
and COMPLETE only after success. Baseline-specific architecture/output/target rules are defined by the named method in `movecar.py`; the reference YAML is not a claim that DDPG has twin critics. No `.env` is read by the experimental driver;
W&B is disabled. Existing output directories are never silently overwritten.
The campaign snapshots code/configs before submission. GPU work uses Slurm,
not the login node. The verified runtime is the existing scratch scalar venv,
with `LD_LIBRARY_PATH` unset; package versions are recorded instead of modifying
other experiments' environments.

Example individual commands (run within an allocated GPU job):

```bash
python -m analysis_boltzmann.frozen --case asymmetric_1.3_0.08 --seed 0 --out NEW_DIR
python -m analysis_boltzmann.movecar --method optiq --seed 0 --out NEW_DIR
python -m analysis_boltzmann.control --backup boltzmann --seed 0 --out NEW_DIR
python -m analysis_boltzmann.plots --root RUNS_DIR --out FIGURES_DIR
```

`campaign --campaign NEW_DIR` freezes sources and submits the pilot array at
concurrency 2. Health gate requires COMPLETE, finite measurements, complete 1M
MoveCar runs and control-grid discrepancy <1e-3. It generates figures and then
runs the paired/cross-critic landscape stage, then submits seeds5–19 with unchanged settings. The final stage adds the remaining cross-critic figures. No gate depends on OptiQ winning.
A failed array or failed gate prevents expansion and leaves logs for diagnosis.

## Verification

`python -m analysis_boltzmann.tests`: constant-Q normalization, separability,
quadrature convergence, correct local-support IS limit, K=1 identity, MoveCar
truncation semantics, NumPy/JAX energy parity. Standalone unittest avoids the
repository's conftest online W&B initialization.

`smoke` checks all five learners, actual updates, serialization/reload, landscape
and figure generation. `control_smoke` checks all control arms, learned-Q
estimates and the 2D coverage initialization. Smoke horizons/update budgets are
explicitly reduced and never included in scientific result aggregation.

Sources: [SD2/SD3 paper](https://papers.nips.cc/paper/2020/file/884d247c6f65a96a7da4d1105d584ddd-Paper.pdf),
[supplement](https://papers.nips.cc/paper_files/paper/2020/file/884d247c6f65a96a7da4d1105d584ddd-Supplemental.pdf),
[official SD3 implementation](https://github.com/ling-pan/SD3/blob/master/SD3.py).
