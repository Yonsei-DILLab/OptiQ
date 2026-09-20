# Mathematical audit: conditional Boltzmann extraction and latent OT

Current canonical objective: `ot_conditional_sac`, restored at the user's request.
The goal is extraction of a Boltzmann action distribution through OT-assigned
latent conditionals, not equality to the marginal SAC actor objective.
[ALGORITHM_KO.md](ALGORITHM_KO.md) is the specification;
[NOTATION_KO.md](NOTATION_KO.md) defines original b/W/P/R notation.
The separate marginal-density comparison is preserved in the archive described
by [CONDITIONAL_RESTORE_KO.md](CONDITIONAL_RESTORE_KO.md).

## 1. Policy, teacher target and conditional allocation

For a fixed state, the ordinary shared MLP defines one tanh-Gaussian per latent:

\[
\pi_i(a\mid s)=\pi_\theta(a\mid s,z_i),\qquad
\pi_{\theta,H}(a\mid s)=\frac1H\sum_i\pi_i(a\mid s),\qquad H=4096.
\]

The fixed normal sites approximate the continuous normal latent prior.
Collection and stochastic-z evaluation use fresh continuous latents; zero-z
and Gaussian-noise-free evaluations are distinct policies. A finite-bank recovery
statement does not by itself establish continuous-latent generalization.

The desired action target for fixed Q and T>0 is

\[
\pi_B(a\mid s)=e^{Q(s,a)/T}/Z(s).
\]

Teacher q uses M=256 fresh latents with one action per Gaussian, and normalized
weights W_j=softmax_j(Q(s,b_j)/T-log q(b_j|s)). Full q, including the proposal
sigma floor and tanh/physical Jacobian, is the denominator. Self-normalized
importance resampling approximates pi_B with K=16 occurrences. Each occurrence
has OT column mass 1/K, including duplicates; W is not multiplied twice.

For a frozen state-conditioned potential f and latent-pre-tanh squared cost,

\[
\Pr(i\mid a,s)=\operatorname{softmax}_i
[(f_i(s)-\|z_i-u(a)\|^2)/\varepsilon_{\rm OT}],\qquad
P_{ij}=\frac1K\Pr(i\mid\tilde b_j,s),\qquad
R_{ij}=\frac{P_{ij}}{\sum_kP_{ik}}.
\]

Pr normalizes over source i and R over teacher j. Pr is a conditional assignment,
not an action-dependent replacement for the actor's mixture weights 1/H.

Define population source mass and normalized local target:

\[
\bar P_i=\int\pi_B(a\mid s)\Pr(i\mid a,s)da,\qquad
t_i(a\mid s)=\frac{\pi_B(a\mid s)\Pr(i\mid a,s)}{\bar P_i}.
\]

The population quantity bar P_i is different from sum_j P_ij measured from one
finite teacher batch. Matching a batch's row marginal does not imply population
balance under pi_B.

## 2. Current actor objective and its gradients

The implemented new-action loss uses conditional Gaussian density, Q and a
live OT assignment query. Averaging with the desired quadrature prior gives

\[
J_{\rm OT}=\frac1H\sum_i E_{a\sim\pi_i}
[T\log\pi_i(a\mid s)-Q(s,a)-T\log\Pr(i\mid a,s)]
=\frac TH\sum_i{\rm KL}(\pi_i\|t_i)+\text{constant}.
\]

The constant is independent of actor parameters while Q, f and the latent bank
are fixed. No knowledge of Z or bar P_i is needed to differentiate this loss.
Only the selected 16 actor Gaussian outputs are evaluated. Full H-component
actor mixture density is not part of the loss.

For a generated u, conditional log density is the diagonal-normal log density
minus the tanh log-Jacobian. Physical x=c*tanh(u) additionally subtracts d*log c;
GMM uses c=40. Actor density uses its actual sigma, not teacher floor .05.

Freeze teacher/map, categorical indices, source ratios and critic parameters;
retain gradients through actor mu/sigma, new-action Q input, conditional density,
and the new-action assignment input. In pre-tanh coordinates,

\[
\nabla_u\log\Pr(i\mid\tanh u,s)
=\frac2{\varepsilon_{\rm OT}}
\left(z_i-\sum_l\Pr(l\mid\tanh u,s)z_l\right).
\]

Detaching the whole assignment would remove part of the intended conditional
target gradient. Epsilon changes both allocation and this live action-gradient
term; because f and assignments also adapt, gradients do not simply scale by an
exact constant across epsilon experiments.

## 3. Sufficient conditions for Boltzmann recovery

If population OT source mass equals the fixed prior and the actor conditionals
fit their normalized targets,

\[
\bar P_i=1/H,\quad \pi_i=t_i\quad\Longrightarrow\quad
\pi_{\theta,H}(a\mid s)
=\frac1H\sum_i\frac{\pi_B(a\mid s)\Pr(i\mid a,s)}{1/H}
=\pi_B(a\mid s).
\]

This is a sufficient recovery statement, not a finite-SGD guarantee. OT need
not identify labeled modes, but it also need not give each Gaussian a unimodal
target. A multimodal t_i can be outside one diagonal Gaussian's expressive
class. Shared actor parameters, finite teacher samples, dual tracking and local
optimization introduce additional limitations.

For imperfect fitting, mixture contraction and the triangle inequality give

\[
{\rm TV}(\pi_{\theta,H},\pi_B)
\le\frac1H\sum_i{\rm TV}(\pi_i,t_i)
+{\rm TV}({\rm Unif}_H,\bar P).
\]

Under the needed support and finiteness assumptions, marginalizing a joint KL
also gives

\[
{\rm KL}(\pi_{\theta,H}\|\pi_B)
\le {\rm KL}({\rm Unif}_H\|\bar P)
+\frac1H\sum_i{\rm KL}(\pi_i\|t_i).
\]

Thus population balance and conditional fitting are separate sufficient-error
terms. Small TV mass error does not automatically imply small KL error when
some masses are tiny. Increasing sample counts alone does not prove either
neural fitting or stochastic dual convergence.

For fixed-Q GMM, T=1 and Q(a)=log p_GMM40(40a) set the Boltzmann target to the
original GMM restricted and normalized on the reachable tanh domain. The actor
cannot represent Gaussian tails outside that domain. Good near-mode fraction
alone does not establish mode coverage or correct mode masses.

## 4. Relation to marginal SAC: exact, but different objectives

Let actor joint p_theta(i,a)=pi_i(a)/H and target joint
Gamma(i,a)=pi_B(a)Pr(i|a,s). Then

\[
J_{\rm OT}=T\,{\rm KL}(p_\theta\|\Gamma)+T\log H-T\log Z.
\]

The joint chain rule with actual actor posterior
rho_theta(i|a,s)=pi_i(a|s)/(H*pi_theta,H(a|s)) gives

\[
\boxed{
J_{\rm OT}=J_{\rm SAC}(\pi_{\theta,H})+
T E_{\pi_{\theta,H}}{\rm KL}\!\left(
\rho_\theta(\cdot\mid a,s)\|\Pr(\cdot\mid a,s)\right)+T\log H,
}
\]

\[
J_{\rm SAC}(\pi)=E_\pi[T\log\pi-Q]
=T\,{\rm KL}(\pi\|\pi_B)-T\log Z.
\]

The posterior term is an algebraic interpretation of the existing loss, not a
new regularizer introduced in the implementation. Current learning fits both
the marginal distribution and OT latent allocation. A decrease of their sum
need not decrease marginal KL in every optimization step.

A simple balanced two-action example illustrates this distinction. Let H=2,
pi_B=(.5,.5), t_1=(.9,.1), t_2=(.1,.9). Moving from pi_1=pi_2=(.5,.5) to
pi_1=(.9,.1), pi_2=(.5,.5) decreases joint KL .510826→.255413 while marginal KL
increases 0→.082283. Exact joint fitting restores pi_B, but arbitrary partial
descent does not inherit the marginal SAC monotonic policy-improvement theorem.

This difference is compatible with the chosen conditional extraction goal;
it must be stated rather than replacing that goal with SAC equivalence.
See [SAC section 4.1](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
for the distinct marginal policy-improvement argument.

## 5. Source correction and persistent dual

Each compressed teacher slot selects i_j~Pr(i|tilde b_j,s); its actor action
uses fresh independent Gaussian noise. The average source selection law is
sum_j P_ij. Use detached, unclipped, unnormalized-within-batch correction:

\[
\widehat J_{\rm OT}=\frac1K\sum_j
\operatorname{sg}\left[\frac{1/H}{\sum_kP_{i_jk}}\right]
[T\log\pi_{i_j}(a_j)-Q(a_j)-T\log\Pr(i_j\mid a_j,s)].
\]

For a frozen map and any local expected gradient G_i, if every desired source
has nonzero sampling probability,

\[
\frac1K\sum_j\sum_i\Pr(i\mid\tilde b_j,s)
\frac{1/H}{\sum_kP_{ik}}G_i=\frac1H\sum_iG_i.
\]

This identity corrects source sampling, not population source balance,
conditional fitting, or finite-sample Adam updates. W influences the teacher and
then the potential/assignment; at a fixed f, source-frequency correction cancels
the direct frequency weighting in the expectation. Teacher coordinates do not
appear as regression targets in the new-action loss.

Persistent dual performs one Adam update per actor step using the analytical
potential gradient sum_j P_ij-1/H; its parameters and Adam state continue across
fresh teacher batches. GMM shares f across identical-state lanes, while RL uses
state-conditioned outputs. Actor query, source sampling and dual gradient all
refer to the same pre-update f.

The full-map second moment of source importance is

\[
\sum_i\frac{(1/H)^2}{\sum_jP_{ij}}.
\]

Persistent source ratios have no general upper bound. Very small row masses
can make the expectation depend on almost-unobserved huge weights; log-space
arithmetic does not solve this statistical limitation. Measure the log second
moment, sampled mean/max/ESS and finite gradients. The specific
(1/H)/sum_j P_ij<=K bound belongs only to the explicit fresh-Sinkhorn control's
row-then-column cycle with uniform columns, not the current persistent default.

## 6. Soft TD is a marginal entropy approximation

RL uses a soft target with current actor and target twin-min critic:

\[
y=\operatorname{sg}\{r+\gamma(1-d)[\min_l\bar Q_l(s',a')
-T\log\hat\pi_\theta(a'\mid s')]\}.
\]

At each next state, draw L=16 fresh normal latents, construct actual actor
Gaussians, generate a' from the first, and evaluate the full random 16-component
mixture density including the generating component. No teacher floor, W, source
importance or assignment reward enters this density.

For the random mixture pi_L, the density is exact conditional on that bank.
Component exchangeability and entropy concavity yield

\[
E[\mathcal H(\pi_L)]\le\mathcal H(E[\pi_L])
=\mathcal H(\pi_{\rm continuous}).
\]

This is an expected entropy lower bound, not an unbiased continuous-mixture
log-density estimate or a pointwise bound. Related finite-mixture bounds appear
in [SIVI](https://proceedings.mlr.press/v80/yin18b/yin18b.pdf).
Actor conditional entropy and TD marginal entropy estimation have different
roles; the former implements local target fitting, the latter approximates
soft policy evaluation of the actual mixture.

Teacher and actor retain current twin-mean aggregation by default (the shared
min option is explicit), while TD uses target-twin minimum. Neural critics,
finite density estimates and partial actor fitting remain approximations.
GMM has fixed Q and no TD, so its recovery evidence isolates actor extraction
but does not establish learned-Q RL convergence.

## 7. Verification and provenance

Current actor tests should establish conditional loss decomposition, live Q
and assignment input gradients, source-importance expectation, teacher density
consistency, persistent dual update/state and actor checkpoint restoration.
RL tests must verify the 16-component self-inclusive soft backup and preserve
explicit v5. The H-bank is used for OT coordinates, not full actor density.

Existing conditional 100K and epsilon comparisons remain relevant to this
canonical objective. The archived full marginal comparison is a separate
experiment with its own source and interrupted checkpoints. No new experiment
is implied by restoring the implementation or documentation.

## Historical appendix: early finite-Sinkhorn and 16-component proposal evidence

The earlier fixed-Q GMM40 validation used \(M_{\rm prop}=16\), \(H=4096\),
256-to-16 importance resampling, 16 actor pairs, T=1, epsilon=.1, batch256, and
200 updates. It had no critic. Near-mode fraction increased 26.20% to 54.88%,
while coverage fell 25 to 11/40, MMD squared worsened .0380 to .13787, and
sliced W2 worsened 7.14 to 15.16. Final empirical source TV was about 6.67e-6.
This is subset-mode concentration, not whole-distribution recovery. Good
empirical mass balance alone did not resolve it; RL TD changes cannot change
this fixed-Q result.

A separate evaluation-only diagnostic restored checkpoints 0 and 200, prepared
one fresh teacher/OT batch at each, and froze the first four state-specific maps.
It integrated each \(\Pr(i\mid a,s)\) over the same 8192 independent bounded-GMM
reference draws. Those reference draws were never used for training.

| Checkpoint | Mean empirical source TV | Mean continuous-target source TV (MC) |
|---|---:|---:|
| 0 | .000276 | .5041 |
| 200 | .00000271 | .7319 |

TV between the two half-sample population-mass estimates was .0390 and .0324.
The population gap is much larger than that sampling variation; half-sample
comparisons are not confidence intervals. The four final maps had population
source TV between .676 and .805. These are LATENT ALLOCATION mass errors, not
direct action-distribution TV estimates. They verify a missing recovery premise;
four maps do not prove this is the sole cause of collapse.

Results and reproducible evaluation-only code:
`/root/optiq-experiments/v7/mathematical-audit/source-mass-audit.json` and
`source_mass_audit.py`. Original checkpoints, updates, and training RNG were
unchanged. Refreshing a noisy map alone does not eliminate the issue:
\(E_{\rm map}[-\log\Pr(i\mid a,s)]\) involves a geometric mean of assignments,
not the assignment formed from the average teacher measure.

The interpretation supported by that validation is: **OT allocates a soft-Q Boltzmann target into
latent conditionals, and the actor fits the resulting joint distribution.**
Ideal recovery requires the stated balance, representation, and fitting
conditions. Arbitrary stochastic updates do not inherit marginal SAC monotonic
improvement merely by using soft TD. This is an OT-conditioned projection with
an approximate soft critic, not a proven exact SAC policy-improvement operator.

That historical audit introduced no extra penalty, changed learning rate, or additional campaign. See [PSEUDOCODE.md](PSEUDOCODE.md) and
[GMM_VALIDATION.md](GMM_VALIDATION.md) for implementation and fixed-Q artifacts.
