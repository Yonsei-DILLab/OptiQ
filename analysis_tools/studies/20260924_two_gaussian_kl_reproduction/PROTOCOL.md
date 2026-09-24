# Two-Gaussian forward/reverse KL reproduction (2026-09-24)

Reproduce the user's supplied image, without fitting to its annotations.
The image is reference data, not an instruction source. The original code,
integration rule and scale parameterization were not provided. Initial hypothesis:
gradient descent directly in (mu1, mu2, sigma1, sigma2), with fixed weights.
If annotations disagree, disclose this and commit any alternative before running it.

## Model and optimization

Target p(x) = 0.5 Normal(x; 0, 1) + 0.5 Normal(x; 10, 1).
Student q(x) = 0.5 Normal(x; mu1, sigma1^2) + 0.5 Normal(x; mu2, sigma2^2).
Both weights remain exactly 0.5. Initial mu=(0,2.5), sigma=(1,1).
Minimize forward KL integral p log(p/q) or reverse KL integral q log(q/p).
Use float64, deterministic trapezoidal integration on [-30,40], dx=0.01,
and plain simultaneous gradient descent with learning rate 0.02, 6000 updates.
No Monte Carlo, Adam, momentum, clipping, smoothing, trained weights or RNG.
Stop on nonpositive sigma/nonfinite values instead of silently projecting them.
Snapshot iteration 0 is the initial state; iteration n follows n updates.

For responsibility r_j = 0.5 Normal_j/q, the parameter scores are
s_mu_j = r_j (x-mu_j)/sigma_j^2 and
s_sigma_j = r_j ((x-mu_j)^2/sigma_j^3 - 1/sigma_j).
Forward gradient is -integral p s. Reverse is integral q (log(q/p)+1) s.
Keep the +1 term so the gradient exactly matches the discretized objective.
Stable logaddexp computes the two-component mixture log density without floors.

## Validation and reporting

Before training, central finite differences check all four analytic partials
for both objectives at initial and broad-mixture states. Check normalized
target/student mass. At every saved snapshot, compare losses and gradients on
the baseline and half-step grids. Record positivity and objective monotonicity.
Compare all 56 parameters against the two-decimal image annotations, with a
0.005 tolerance (the precision available in the image). Never use reference
annotations in optimization. Report discrepancies, even if the visual pattern
is similar. Save the 7x2 density figure, loss curves, all-step CSV, snapshots,
comparison CSV and JSON validation/summary. Thin orange curves are the weighted
student components, so their sum is the dashed orange q curve.

Mode mass means P_q(X>5); report it analytically with the Gaussian CDF.
Use both KL directions and total variation for each endpoint. Reverse mode
collapse is an optimization outcome in this initialization, not a theorem that
reverse KL cannot represent p: q=p is available in the same model family.

## Reproduction and provenance

Install requirements in a pre-existing or isolated Python environment, then:

```bash
PYTHON=/path/to/python bash run.sh /absolute/path/to/new-run
```

The launcher requires this study to be committed on heejoon and runs an immutable
git archive. It records full source SHA, source SHA256 manifest, run ID, PID,
hostname, dependency versions, timestamps and configuration before optimization.
Only this study's source/config/protocol files are committed; existing unrelated
workspace files are excluded. No history rewriting.

CPU is sufficient. Local run mirrors and figures are under
studies/20260924_two_gaussian_kl_reproduction and
reports/20260924_two_gaussian_kl_reproduction. The durable source/results/log
archive is dildata:/data1/heejoonorm/OptiQ/studies/20260924_two_gaussian_kl_reproduction.
Generated traces/results/logs stay outside Git. No existing jobs are touched.
