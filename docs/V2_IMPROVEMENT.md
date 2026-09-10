# v2 improvement investigation

## Stop decision, 2026-09-10

All four original v2 runs were stopped at about 154k-155k environment steps after
preserving their 150k actor and critic checkpoints. At 140k, returns were
289/281/296/237 for seeds 0/1/2/3, versus 1123/1452/818/722 in historical OptiQ.
The v2 curves were flat or declining over 100k-140k. The historical runs used
10% additional uniform exploration, so this is a resource-allocation decision,
not a controlled causal comparison. Logs and W&B runs are retained and annotated.
Machine-readable evidence: `outputs/v2_improvement/early_stop_evidence.json`.

ESS remained around 28-32/64 while the entropy contribution to the TD target was
around -5, comparable in magnitude to the per-step environment reward. High ESS
therefore did not establish good policy learning.

## Flat-Q diagnosis

`scripts/diagnose_v2_projection.py` fixes Q to zero on 32 stored Humanoid states,
keeps the 17-dimensional action space, and runs 1001 actor updates. The exact
Boltzmann target is uniform on the action box regardless of temperature.

| Projection | Initial entropy estimate | Final entropy estimate | Final mean conditional sigma | Final ESS / 64 |
|---|---:|---:|---:|---:|
| argmax + pointwise MSE | 8.38 | -2.43 | .223 | 32.43 |
| full OT conditional Gaussian NLL | 8.38 | 11.48 | .951 | 3.52 |

The uniform entropy maximum in normalized coordinates is 17*log(2)=11.78 nat.
This isolates a projection failure even without critic errors: high ESS and a
valid transport plan do not imply that a hard argmax student retains its target
distribution. In the MSE run, converting the rows to argmax assignments changed
the column masses by mean TV distance .577 at the final update.

The NLL loss uses the full stopped OT row. Its teacher pre-tanh mean and variance
are sufficient statistics for the Gaussian conditional cross entropy. This
retains the row's variance instead of fitting a single point. The teacher, Q,
weights and plan remain stopped; there are no actor gradients through Q, no
additional actor entropy bonus, and no added uniform behavior exploration.
The tanh Jacobian of fixed teacher samples is constant with respect to student
parameters and need not enter this loss. This is a different projection loss;
it is not claimed to optimize exact Wasserstein distance or reverse KL.

## Bounded screening experiment

Four fresh seed-0 Humanoid runs, maximum 100k steps each, compare:

1. Original `mujoco_setting` OptiQ, T=.25, anchor=true, extra uniform probability 0.
2. v2 conditional OT NLL, T=.5.
3. v2 conditional OT NLL, T=.1.
4. v2 original pointwise MSE, T=.1 (temperature-only control).

All use three stochastic evaluation episodes every 5k steps for screening,
25k checkpoints, 256x3 networks, batch 256, UTD1, 5k warmup, and gradient norm2.
The v2 runs retain 16x64 OT, h=.8, beta1 and M16 entropy components. At 50k and
75k, review the last three evaluation checkpoints. A variant with less than half
the matched reference's return and no improving trend is eligible for early
termination after preserving a checkpoint. This is a pilot selection rule, not
a statistical claim of superiority. Only promising settings advance to the
four-seed, ten-episode evaluation protocol.

W&B: `OptiQ/optiq_mujoco_v2_screen`. Supervisor definition:
`deploy/supervisor/optiq-v2-screen.conf`. No git push is performed.

## Policy-improvement requirement

Neither the original argmax-MSE nor the new conditional NLL alone provides a
monotonic soft-policy-improvement guarantee. A lower projection loss is not an
acceptance certificate. For a fixed temperature, define

`F_Q(pi,s) = E_{a~pi}[Q(s,a)] + T*H(pi(.|s))`.

With the exact old-policy soft Q, the sufficient condition is
`F_Q(pi_new,s) >= F_Q(pi_old,s)` at every relevant state. The new-policy Bellman
operator is then monotone and contractive, giving nondecreasing soft values.
This condition does not require taking gradients through Q or using KL as the
proposal loss. It does require evaluating the proposed policy, rather than
assuming that OT distillation has improved it.

For an approximate critic satisfying a *verified uniform* error bound epsilon,
the estimated F gap must exceed 2*epsilon. If new entropy has a lower bound and
old entropy has an upper bound, use those sides in the acceptance gap. Sampling
confidence errors must also be subtracted. A replay average or twin-critic
disagreement is not a certificate of either global state coverage or uniform
critic error. Those are unresolved for the actual Humanoid runs. Finite-M IDAC
entropy is a lower bound in expectation; comparing two lower bounds does not
certify that the true entropies improved.

The appropriate next step is to evaluate candidate updates using these soft
value conditions and explicitly distinguish a practical sampled acceptance
check from a theorem under exact evaluation/error-bound assumptions. There is
currently no claim that practical Humanoid improvement is guaranteed.

References: [SAC soft policy iteration, Appendix B](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
and [IDAC entropy estimation](https://arxiv.org/html/2007.06159). The proposed OT
projection and the approximate-critic acceptance margin above are our analysis,
not claims attributed to the original IDAC algorithm.
