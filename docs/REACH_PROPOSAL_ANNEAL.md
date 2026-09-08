# Reach candidate KDE annealing comparison

This comparison starts from `critic-dime-no-anchor` commit
`f8f622b09c760e011ef956aa9bdc30cdd336d8e2`, in the isolated branch
`critic-dime-no-anchor-temperature`. It broadens candidate generation early in
training and restores the baseline kernel later. The Q/OT weight temperature
remains **0.25** throughout training.

## Matched protocol

Two fresh 1,000,000-step `myoHandReachRandom-v0` runs use seeds **0 and 2**.
These seeds had the lower mean returns over the last ten baseline evaluations
(100 episodes per seed), rather than one noisy final evaluation:

| Seed | Mean return | Success rate | Baseline W&B run |
| --- | ---: | ---: | --- |
| 0 | 693.9034 | 0.90 | [l2y53ej6](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/l2y53ej6) |
| 2 | 703.1609 | 0.91 | [rumairio](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/rumairio) |
| 1 | 708.2034 | 0.91 | [ms4wa6ip](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/ms4wa6ip) |

This is a selected-seed exploratory comparison, not an unbiased estimate of an
improvement over all seeds. Compare returns, success, ESS, weight concentration,
critic disagreement and candidate Q diagnostics on matched environment steps.

The experiment inherits all baseline settings, including 16 policy centers,
four random candidates per component (16 x 64 OT), no candidate anchors,
density beta 0.1, Sinkhorn epsilon 0.05, 30 iterations, argmax transport targets,
pointwise MSE, DIME critic support [-3600, 3600], UTD 2, learning starts at 5,000,
and TD action noise std 0.2 / clip 0.5. Evaluation remains every 5,000 steps with
10 stochastic episodes, diagnostics every 5,000 and checkpoints every 50,000.
Actor and critic gradient calculations are inherited unchanged.

## Proposal schedule

For environment step `t`, the dimensionless kernel variance multiplier is:

```text
kappa(t) = 20                                  t <= 50,000
           20 - 19 * (t - 50,000) / 150,000     50,000 < t < 200,000
           1                                   t >= 200,000
proposal_std(t)  = 0.2 * sqrt(kappa(t))
proposal_clip(t) = 0.5 * sqrt(kappa(t))
```

Initially std is approximately 0.8944 and the local clip radius is 2.2361.
The action box remains [-1, 1] per coordinate. Thus early kernels can cover the
whole action box; samples are drawn from truncated normals, not Gaussian draws
clamped onto the boundary. Both std and local support shrink to their baseline
values by 200,000 steps. Only the kernel parameters return to baseline; the
learned actor, critic and replay history can remain different.

The KDE is defined from policy centers before drawing candidates, with four
draws from each component. Sampling and mixture log-density evaluation use the
same scheduled std, support and truncation normalizers. The actor centers are
not inserted as candidates. The schedule does not directly modify environment
action noise or the TD target action noise.

`train/proposal_temperature`, `train/proposal_std`, `train/proposal_clip` and
`train/temperature` record the actual schedule and fixed Q temperature.

## Interpretation and PITA

This is kernel broadening, not exact global KDE power tempering: widening each
component does not generally equal raising the whole mixture density to a
fractional power. The factor 20 is an experimental choice, not a Kelvin value
copied from a molecular system. It multiplies the underlying Gaussian variance;
the variance after action-space truncation is not multiplied by exactly 20.

With proposal density `g_t`, candidate weights are proportional to
`exp(Q / 0.25) / g_t^0.1`. At the idealized weighted-candidate measure level,
the effective density on proposal support is therefore proportional to
`exp(Q / 0.25) * g_t^0.9`. Holding Q temperature fixed preserves Q discrimination,
but partial density correction leaves proposal dependence. The finite OT plan,
argmax assignment and actor distillation are further approximations; this does
not guarantee exact sampling of a fixed Boltzmann density.

[PITA](https://arxiv.org/html/2506.16471v2) progressively changes intermediate
Boltzmann targets, combining diffusion models with Feynman-Kac importance
weighting and SMC resampling to reach a lower final temperature. Its main
peptide experiments use 1200 K to 300 K. This comparison borrows the motivation
of early broad exploration, but does not implement PITA or its correction
scheme. Raising our Q/OT temperature would be a separate intervention on the
actor's learning target.

## Run

Use the existing `/workspace/.venv-optiq-no-anchor` environment and an ignored
`.env` configured from `.env.example`.

```bash
bash scripts/run_reach_proposal_anneal.sh --list
CUDA_VISIBLE_DEVICES=2 bash scripts/run_reach_proposal_anneal.sh --task 0
CUDA_VISIBLE_DEVICES=3 bash scripts/run_reach_proposal_anneal.sh --task 1
```

For managed runs, `scripts/supervisor_reach_proposal.conf.example` assigns seed
0 to GPU 2 and seed 2 to GPU 3. It disables automatic restarts so completed runs
are not replayed. Existing baseline workers on GPUs 0 and 1 use their original
worktree. Outputs are stored under `outputs/optiq_dime_reach_proposal_anneal`,
with a separate W&B group in the baseline project. Each run saves its resolved
configuration and Git/package/GPU provenance.

## Validation

`tests/test_proposal_schedule.py` checks resolved baseline parameter equality,
schedule boundaries and invalid settings, candidate sampling/density agreement
against an analytical truncated normal, and actual training updates with fixed
Q temperature and TD noise. A short full-network Reach GPU/JIT run additionally
checks the high, decreasing and final proposal scales and W&B logging.

Validation on 2026-09-08: all 49 tests passed. The 12-step GPU run performed 16
updates, recorded all three schedule stages with finite losses and Q temperature
0.25, and saved checkpoints and W&B artifacts:
[validation run](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/8x5isaht).
Its abbreviated schedule and learning warmup are smoke-test overrides only.
