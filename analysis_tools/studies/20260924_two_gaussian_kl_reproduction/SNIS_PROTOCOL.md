# Third column: current-policy sampling and SNIS NLL

User request: add a third column to the existing two-Gaussian figure, training
with sampling and self-normalized importance-weighted NLL. The user now requests
4096 IID mixture samples per update, replacing 256. Seed 0 is fixed before execution; do not
select a seed based on recovery. This is a single illustrative trajectory.

The prior 256-sample run remains frozen at source commit
7c50def448077b06e3282b38c56ffb215e3bfd20. Only samples_per_update changes in
the numerical configuration; use the identical optimizer and sampling code.
Restart from the original initialization, not the previous final parameters.
The new run ID is snis_n4096_seed0_run01. Across 6000 updates it draws 24576000
training samples. The same seed does not imply identical or nested samples
after batch size changes. Preserve prior results and compare endpoints.

Reuse the saved forward/reverse baseline snapshots from source commit
7622a42fa6a7c9c46762da649f6ec8344fd463dd; do not rerun or mutate that run.
Copy its snapshots and provenance into each new run's inputs and hash both.
Target, model, fixed mixture weights, direct mu/sigma parameters, initial state,
learning rate 0.02, 6000 updates and 7 snapshot times are exactly the baseline.
The baseline config.json remains unchanged; extension settings are snis_config.json.

At update t, choose 4096 component indices IID with probabilities (0.5,0.5),
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
is no extra division by batch size after weights have been normalized to sum to 1.
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
