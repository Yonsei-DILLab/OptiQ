# Four-way Direct GMM / OptiQ diagnostic

This is a short, single-seed diagnostic of whether the current Direct GMM
OptiQ learner can keep four symmetric behaviors at one identical state. It
reuses the existing two-dimensional four-goal environment and the production
`OptiQDIME` update; it does not use the older proposal-family learner in
`experiments/four_way_multigoal/proposal_recovery.py`.

## Task and budget

- State and action are 2-D. Every episode starts at `(0, 0)`.
- The four terminal goals are `(±5, 0)` and `(0, ±5)`, with success radius 1
  and a 20-step horizon.
- Preserve the existing dense reward: `-30 ||a||² - min_g ||s' - g||²`,
  with `+10` on goal entry. Goal termination is unchanged.
- Train for exactly 10,000 environment transitions, one environment, seed 0.
  The first 1,000 transitions are uniform-action warm-up; the remaining
  transitions receive one learner update each with batch 256.
- Evaluate from the same origin at 0, 1k, 2k, 5k, and 10k transitions. Use
  40 fixed-origin episodes per mode at interim points and 100 at 10k. Record
  both the full conditional policy (fresh normal latent plus conditional
  Gaussian noise) and random-latent μ-only policy.

## OptiQ profile

Use the committed Direct GMM/TRG `OptiQDIME` code with N=M=64, direct marginal
GMM NLL, β=1 density correction, T=1, scalar twin Q critics, 256x2 GELU
actor/critics, mean-head initialization 1e-4, log σ in [-5,-1] initialized at
-1, γ=.99, critic EMA τ=.005, actor/critic Adam 3e-4, and no DACER or NovelD.
This quick diagnostic changes only task dimensions/geometry, short budget,
single-env collection, and warm-up length. It is not an AntMaze result or a
replacement for a long-run benchmark.

## Outputs and interpretation

Save raw rollout xy arrays, first actions, success/goal ids, origin directional
Q probes, learner diagnostics (including source ESS and Q/density logit scales),
resolved config, and the source commit. The main plot separates actual
goal-reaching frequencies from first-action direction masses. One seed and
10k transitions are exploratory evidence only; entering four directions is
not equivalent to reaching four goals.
