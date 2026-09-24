# Third column: current-policy sampling and SNIS NLL

User request: add a third column to the existing two-Gaussian figure, training
with sampling and self-normalized importance-weighted NLL. The user selected
256 IID mixture samples per update. Seed 0 is fixed before execution; do not
select a seed based on recovery. This is a single illustrative trajectory.

Reuse the saved forward/reverse baseline snapshots from source commit
7622a42fa6a7c9c46762da649f6ec8344fd463dd; do not rerun or mutate that run.
Copy its snapshots and provenance into each new run's inputs and hash both.
Target, model, fixed mixture weights, direct mu/sigma parameters, initial state,
learning rate 0.02, 6000 updates and 7 snapshot times are exactly the baseline.
The baseline config.json remains unchanged; extension settings are snis_config.json.

At update t, choose 256 component indices IID with probabilities (0.5,0.5),
then x_i = mu_{k_i} + sigma_{k_i} epsilon_i, epsilon_i ~ Normal(0,1).
The proposal is the exact current two-component mixture q_t, not a component
density, a target sampler, a uniform proposal or a wide auxiliary proposal.

log w_i = log p(x_i) - log q_t(x_i); wbar = softmax(log w).
Hold x_i and wbar_i fixed while differentiating:

L_t(theta) = -sum_i wbar_i log q_theta(x_i).

The analytic gradient is -sum_i wbar_i s_theta(x_i), where the mixture score
is computed with full mixture responsibilities exactly as in the baseline.
Update all four parameters simultaneously with plain gradient descent.
No derivative flows through samples, proposal density or normalized weights.
Weights are recomputed from the pre-update policy on each fresh batch; there
is no extra division by 256 after weights have been normalized to sum to 1.
Stable log weights prevent numerical underflow; no weight clipping or tempering.
No momentum, Adam, sigma projection, target samples or forced right-mode samples.
Fail explicitly if an update makes a scale nonpositive or a value nonfinite.

Save per-update sampled NLL, ESS=1/sum(wbar^2), maximum weight, proposal sample
right-basin count and range, parameter values and gradient norm. The NLL uses
a new batch every step, so it is not a single fixed objective curve or a true
KL estimate. Report quadrature KL(p||q), KL(q||p), TV and analytic P_q(X>5) at
snapshots using the baseline evaluation routine, with no evaluation RNG.
Density panels plot the exact learned mixture to match the first two columns;
training itself uses sampled actions, and the panel title states batch and seed.

Before optimization, central finite differences validate the frozen-sample,
frozen-weight NLL gradient for all four parameters at two parameter states.
Verify normalized weights, ESS bounds, invariance to target log-normalizer,
uniform weights when p=q, and seeded sampler reproducibility. Validation uses
a separate RNG and cannot advance the training RNG. Record finite-value and
positive-scale checks. Finite-batch SNIS is biased and may fail to discover
missing modes; report the actual outcome, without retrying seeds.

Commit source, configuration, launcher and protocol to heejoon before the run.
Run from the immutable Git archive. Record full source commit, manifest, run ID,
PID, timestamps, environment, baseline input hashes and config. Archive source,
raw traces and results to dildata under the existing study directory. Existing
two-column source snapshots and figures remain available; save the new main
figure as reproduction_snis.png and reproduction_snis.svg.

```bash
PYTHON=/path/to/python bash run_snis.sh /absolute/path/to/new-run /absolute/path/to/direct_sigma_run01/results
```
