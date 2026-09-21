# HalfCheetah raw-energy soft-Bellman campaign, 2026-09-22

## Hypothesis

The raw GMM40 circuit becomes a stable tied policy/Q learner when its Bellman
target uses the exact soft value that the same circuit represents:

    F_theta(s,a) = exp(Q_theta(s,a) / alpha),
    pi_theta(a|s) = F_theta(s,a) / Z_theta(s),
    V_theta(s) = alpha log Z_theta(s),
    y = r + gamma (1-terminal) V_bar_theta(s').

This target preserves the one-forward analytic sampler and introduces no
independent value head, actor or critic. It is the soft-optimality update for a
tied Gibbs policy. Direct's sampled ordinary-TD target is appropriate for its
separate critic and actor, but the 2026-09-22 HalfCheetah diagnostic showed that
the same target can create runaway feedback in a single tied energy circuit.

## One-variable pilot

The baseline is the `gmm_reference` raw-energy run at commit `734b4f6`, stopped
after all seeds had written a 100k checkpoint. Its 100k returns for seeds 0–2
were `-20.81`, `-257.96`, and `29.65`; seed 1's Q scale and gradients diverged.

The 100k pilot keeps raw-energy rank 64 / 129 knots, MSE, unclipped Adam 3e-4,
temperature .25, replay, warmup, batch size, target EMA, update ratio,
environment and evaluation seeds fixed. Only the bootstrap changes from a
sampled delayed Q to the delayed circuit's exact `alpha log Z`. Stability and
`eval/mean_reward` are the evaluation metrics. If all three seeds remain finite,
launch fresh seeds 0–2 for 1M steps and compare final reward and 0–1M AUC with
the completed legacy baseline.

W&B remains offline. Outputs are preserved under the campaign directory and
are not automatically uploaded or synced.
