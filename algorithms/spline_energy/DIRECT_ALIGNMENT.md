# GMM40-reference spline with Direct GMM/TRG Bellman training

## Source audit, 2026-09-22

The private `direct-gmm-trg` branch was fetched and inspected at
`30f4db1cf9929974dacbdbca7c9c5abd5c1bd346`. References below refer to that
commit, not the moving branch name:

- `gmm40/optiq_trg.py:OptiQTRG._update`: a fixed oracle Q, importance logits
  `Q / temperature - log_proposal`, then detached weighted mixture NLL.
- `analysis_tools/experiments/20260920_truncated_mll/optiq_dime/algorithm.py`,
  `update_critic`: `backup_mode=td` samples the current policy, evaluates the
  delayed critics, takes their minimum, and uses squared TD error. There is
  no entropy adjustment in this mode.
- The same file, `update_actor`: the teacher Q is separate from the actor;
  beta=1 retains the full `-log_proposal` density correction.
- `analysis_tools/experiments/20260921_gmm_trg_sweep/PROTOCOL.md` and the
  corresponding `train.py`: ordinary TD, no actor gradient clipping, 64
  components / 64 candidates, scalar twin critic.
- Successful spline reference, unchanged in this branch:
  `benchmarks/gmm40/spline_energy/spline_energy.py` and
  `reference_100k_compare_latest.py:spline_update` (64 roots / 129 knots).

## 1. Restore the successful energy parameterization

The GMM40 circuit learns unnormalized roots and leaves:

    F_theta(s,a) = sum_r exp(h_r(s)) product_d f_rd(a_d|s),
    I_rd(s) = integral_-1^1 f_rd(x|s) dx,
    Z_theta(s) = sum_r exp(h_r(s)) product_d I_rd(s),
    Q_theta(s,a) = alpha log F_theta(s,a),
    pi_theta(a|s) = F_theta(s,a) / Z_theta(s).

The leaf integral is the exact trapezoid sum because f is positive piecewise
linear. Sampling selects a root with probability exp(h_r) product_d I_rd / Z
and uses the same analytic leaf inverse CDF as the successful reference.
There is one state-network forward pass and no action optimization at inference.

`raw_energy.py:ConditionalRawEnergyCircuit` outputs h and the raw leaf log
heights. It has **no independently predicted value head**. The legacy
`V_theta + alpha log pi_theta` parameterization is mathematically valid and can
represent the same family; the restoration changes parameterization and its
optimization geometry. It is not a proof that the legacy family was incapable.

The new profile restores rank 64 and 129 knots. At initialization, zero head
kernels make the conditional model exactly equal to the state-free reference
at every state in 2D (same seed, centers, widths, roots and leaf heights). For
other dimensions, a random box-center layout replaces the 2D Cartesian grid;
the reference width .25 is retained. This extension is explicit, not a claim
that GMM40 empirically validated 8D or 17D conditional control.

## 2. How GMM40 energy fitting relates to Direct's likelihood

For a fixed oracle energy F*, sampled actions a~b and known density b, the
GMM40 generalized-KL loss, omitting target-only terms, is

    L_energy = Z_theta - E_b[(F*(a)/b(a)) log F_theta(a)].       (1)

Writing F_theta=Z_theta*pi_theta and Z*=integral F* gives

    L_energy = Z_theta - Z* log Z_theta
               - Z* E_pi*[log pi_theta].                     (2)

In an unrestricted mass parameter, the optimum is Z_theta=Z*. The remaining
shape term is forward KL, up to constants and scale. Direct approximates the
normalized shape term using detached SNIS weights

    w_i = softmax_i(Q_teacher(a_i)/alpha - log b(a_i)),
    L_Direct = -sum_i w_i log pi_theta(a_i).                   (3)

Its finite-sample self-normalization is not the original unbiased estimator
in (1). `fixed_q_forward_energy_loss` retains (1), including the exact Z term
and unnormalized importance weights, for a fixed queryable oracle. The original
GMM40 files and measurements are not replaced.

### Why not paste (3) into the tied spline and call it RL?

If Q_teacher is the same tied energy and b=pi_theta, then

    Q_theta/alpha - log pi_theta = log Z_theta,               (4)

which is constant across actions. The population target is the current policy.
With a defensive uniform mixture as proposal, correct importance weighting
still recovers the same current policy. Finite sampling noise or proposal
changes do not create a principled improvement signal. Direct avoids this
circularity by having a separately trained critic. We preserve one tied energy
instead, so the improvement signal must enter through Bellman learning.

## 3. Correct the Bellman objective for the Direct-aligned profile

Direct's ordinary TD does not identify a log partition with policy value.
For the spline's ordinary reward Q, use

    a' ~ pi_theta(.|s'),
    y = stop_gradient[r + gamma (1-terminal) Q_bar_theta(s',a')],
    L_TD = E[(Q_theta(s,a)-y)^2].                             (5)

The current circuit supplies the action; the EMA circuit evaluates its Q.
No actor gradient is propagated through the sampled target. True terminations
zero the bootstrap, while time-limit truncations retain it. MSE has conditional
minimizer E[y|s,a], so target noise is not converted into an exponential risk
criterion as it would be by exponentiating sampled TD targets.

The identity alpha log Z=alpha log integral exp(Q/alpha) still holds, but in
ordinary TD this quantity is **not** E_pi[Q]. In particular,

    E_pi[Q] = alpha log Z + alpha E_pi[log pi]
            = alpha log Z - alpha H(pi).                    (6)

The old `r+gamma*alpha_log_Z_next` is a valid *soft* Bellman target. It differs
from (5) and was not a matched objective to Direct's ordinary TD. In `td` mode
the trainer logs `alpha_log_partition_mean`, not a misleading ordinary V.

For a fixed frozen policy, the ordinary evaluation Bellman operator is a gamma
contraction on bounded Q; MSE sample targets estimate that operator. Once pi
changes with the tied Q, this statement alone does not prove contraction of the
combined update or neural convergence. The soft-optimality bounds in THEORY.md
are not claimed for this ordinary-TD profile.

## 4. Deliberate differences and implementation scope

The new `--profile gmm_reference` selects raw energy, 64 roots / 129 knots,
ordinary TD and MSE. It uses unclipped Adam 3e-4, as both original GMM40 and
the inspected Direct profile do. Policy temperature remains .25; there is no
new regression temperature, exponential TD loss, or tail threshold.

This is a combined reference restoration, not a one-variable ablation against
the old port. A future reward difference cannot be attributed to one change.

Direct uses twin critics and a separate actor; this spline keeps a single
energy circuit and its EMA. It therefore has no clipped-double-Q protection.
Direct's additional mean-only evaluation removes conditional Gaussian noise;
the spline continues to evaluate its native stochastic policy. A fair numerical
comparison must use matched stochastic evaluation, not silently compare these
different action laws. No Gaussian, tanh squash, separate actor, or extra
critic is added here.

Existing Huber and failed relative-energy variants retain their legacy profile
and frozen outputs. New runs should explicitly select the restored profile:

```bash
cd analysis_tools/experiments/20260921_mujoco_spline_energy
python train.py --profile gmm_reference --env Ant-v4 --seed 0 \
  --total-steps 100000 --output /path/to/new/local/run --wandb-mode offline --require-gpu
```

Targeted validation compares the actual successful reference's Q, log Z,
density, sampler and generalized-KL factor gradients with the restored model;
checks (4); checks the exact ordinary-TD/MSE target and stopped target gradient;
and runs the actual update on a small deterministic terminal problem. These
checks establish code/formula agreement, not MuJoCo learning performance.

## Validation measured on 2026-09-22

Code `4290a6a3b76b91705f981038d7f433d85f800db3`, CPU JAX, unchanged existing
Python environment:

- Original rank64/129-knot GMM40 versus restored raw factors: maximum
  generalized-KL gradient difference `7.45e-9`; maximum coupled sampler
  action difference `5.96e-8`. Q, partition and density checks passed.
- Self-teacher logit range `4.77e-7`, consistent with the constant in (4).
- The implemented plain-TD target/MSE matched the explicit formula; target
  parameter gradient was zero.
- A small terminal contextual regression task decreased MSE from `0.63438`
  to `0.30393` over 192 updates. This is a gradient-path check, not an RL score.

Receipts and the inspected Direct source files are local under
`/scratch2/gsmin2024/research/optiq_spline_gmm_reference_20260922/`.
The initial contextual check used all-zero observations, which together with
zero output kernels exercised only state-free biases and did not pass its
50% loss-reduction threshold. The nonzero-context correction retained that
threshold, learning rate, and update budget; the first outcome is preserved
in `checks/validation_attempt1.json`.

No new MuJoCo reward measurement or improvement claim accompanies this change.
