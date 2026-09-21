# From GMM40 energy fitting to soft Bellman fitting

Status: proposed loss-only extension, 2026-09-22. The proofs below are
conditional mathematical results; they are not a proof that neural-network
training converges or that this variant improves MuJoCo reward.

## 1. What is preserved, and what actually changed in the first port

GMM40 fitted a fixed known target density using generalized KL between positive
energies. Its proposal was a half-uniform/half-current-policy mixture, with a
known density and direct oracle evaluations at arbitrary actions. The successful
reference used rank 64, 129 knots, 16,384 target queries/update, and two dimensions.
See `benchmarks/gmm40/spline_energy/reference_100k_compare_latest.py:spline_update`.

The first MuJoCo port introduced state conditioning, reduced rank/knots to 16/33,
replaced the oracle with a bootstrapped target, and replaced energy fitting with
Huber regression. Thus it did not establish that the GMM40 training algorithm
transfers to control unchanged. Sampler correctness is only one of these issues.

This revision changes only the regression loss, behind `--loss-kind
relative_energy`. It does not change the state encoder, positive spline leaves,
root weights, exact integrals, policy temperature, target EMA, sampler, replay,
action bounds, or number of networks. No twin head, new actor, floor density,
reward clipping, or trust-region optimizer is introduced. The frozen 15-run
Huber experiment remains untouched. Rank/knots stay fixed in the paired pilot
to isolate this change; this does not rule out insufficient capacity.

## 2. Setting and exact representation

Use normalized actions A=[-1,1]^D and Lebesgue measure da. Entropy in this note
is defined in those coordinates, as in the existing trainer. With termination,
the next-state kernel is substochastic (or equivalently include a zero-value
terminal state). Write E[m V(s')] with m=1 for nonterminal transitions. True
terminations have m=0; time-limit truncations retain m=1.

Assume alpha>0, 0<=gamma<1, bounded rewards, and bounded Q functions with finite
partition functions. Uniform bounds over all states are assumptions, not facts
verified by replay samples. MuJoCo observation aliases and unbounded state or
reward domains may prevent direct application of these global statements.

The unchanged circuit has

    pi_theta(a|s) = sum_j w_j(s) product_d f_jd(a_d|s),
    Q_theta(s,a)  = V_theta(s) + alpha log pi_theta(a|s).

Each f is positive and has integral one, and sum_j w_j=1. Consequently

    F(Q)(s) := alpha log integral_A exp(Q(s,a)/alpha) da = V_theta(s),
    pi_theta(a|s) = exp((Q_theta(s,a)-F(Q_theta)(s))/alpha).

Proof: factor exp(V/alpha) out of the integral, then use integral pi=1.
This holds for every parameter vector, including badly trained policies. It
is a representation identity, not an optimality certificate. Selecting a root
then inverting the positive linear leaves' CDFs samples this same density with
one state-network forward pass.

## 3. Soft Bellman residual gives conditional performance bounds

Define the ordinary expected soft Bellman operator

    (T Q)(s,a) = E[r + gamma m F(Q)(s') | s,a].

For bounded Q and R, let d=||Q-R||_infinity. Since Q<=R+d,
F(Q)<=F(R)+d, and the reverse inequality follows symmetrically. Thus

    ||F(Q)-F(R)||_infinity <= ||Q-R||_infinity,
    ||T Q-T R||_infinity <= gamma ||Q-R||_infinity.

Banach's theorem gives a unique fixed point Q*. If the TRUE expected residual
satisfies ||Q-T Q||_infinity<=epsilon, then

    ||Q-Q*||_infinity <= epsilon/(1-gamma).                  (1)

Proof: apply the triangle inequality through TQ and TQ*, then rearrange.

Let pi_Q be the exact Gibbs density above, and T^pi the fixed-policy soft
evaluation operator. The Gibbs variational identity gives

    F(Q) = E_pi_Q[Q-alpha log pi_Q],  T^pi_Q Q = T Q.

T^pi_Q is also a gamma contraction, so its fixed point Q^pi_Q obeys
||Q-Q^pi_Q||_infinity<=epsilon/(1-gamma). Applying the triangle inequality
to state values, using the same Gibbs identity for the middle term, gives

    0 <= V*(s)-V^pi_Q(s) <= 2 epsilon/(1-gamma).             (2)

These are entropy-regularized state values, not the undiscounted raw returns
reported by evaluation. Moreover, from

    log(pi_Q/pi*) = (Q-Q* - F(Q)+F(Q*))/alpha

both directions of statewise policy KL are bounded by

    KL(pi_Q || pi*), KL(pi* || pi_Q)
        <= 2 epsilon / (alpha (1-gamma)).                   (3)

If an approximate value-iteration step satisfies
||Q_{k+1}-TQ_k||_infinity<=e_k, then

    ||Q_k-Q*|| <= gamma^k ||Q_0-Q*||
                 + sum_{j=0}^{k-1} gamma^(k-1-j) e_j.       (4)

Neural SGD with replay and a moving EMA target is not exact value iteration.
In particular, small minibatch Huber loss does not establish any premise in
(1)-(4). A target-lag correction is also needed:

    ||Q-T Q|| <= ||Q-[E r+gamma P V_bar]||
                 + gamma ||V_bar-F(Q)||.                   (5)

## 4. Derive the new loss from the GMM40 generalized KL

For positive intensities u and v, generalized KL is

    D_I(u||v) = u log(u/v)-u+v.

GMM40's integrated loss, up to target-only constants, is
integral exp(Q_theta)-integral exp(Q_target) Q_theta.

For a frozen Bellman sample target y=r+gamma m V_bar(s'), let
u=exp(y/eta), v=exp(Q_theta/eta), eta>0. Dividing the pointwise divergence
by the fixed positive target intensity eliminates the absolute energy scale:

    eta^2 D_I(u||v)/u
      = eta^2 [exp((Q_theta-y)/eta) - (Q_theta-y)/eta - 1].   (6)

Neither exp(Q_theta/eta) nor exp(y/eta) needs to be evaluated separately.
For eta=alpha, these are precisely the policy's unnormalized energies; another
eta changes the regression geometry but not the deterministic pointwise zero.
Eta is NOT a replacement for policy temperature alpha.

This is a deliberate target-relative REWEIGHTING of the GMM40 divergence,
not an unbiased estimator of its original action-integrated objective. We
fit under replay's distribution and do not pretend that its unknown action
density equals one. For a deterministic target with full coverage and
unrestricted realizability, every positive weighting has the same unique
pointwise minimizer Q=y. Under finite capacity, weighting changes the projection.

The prototype uses eta=10 reward units (the existing Huber transition scale)
and alpha=.25, with a local curvature of one: loss=d^2/2+O(d^3/eta).
These are explicit pilot choices, not theoretically optimal hyperparameters.

## 5. Stable tail, fixed point and gradient bound

Exponentiating large TD errors is unsafe. For kappa>0 define

    phi_k(x) = exp(x)-x-1,                                      x<=kappa
             = phi(kappa)+(exp(kappa)-1)(x-kappa)
               + .5 exp(kappa)(x-kappa)^2,                      x>kappa.

Use L=eta^2 phi_k((Q-y)/eta). This is a quadratic continuation, not a hard
clip and not the Maclaurin polynomial used by MXQL. The value, first derivative
and second derivative agree at kappa. Let g=phi_k'. Then

    g(x) = exp(x)-1                        if x<=kappa,
           exp(kappa)-1+exp(kappa)(x-kappa) otherwise.

g is strictly increasing and convex, g(0)=0, and 0<phi_k''<=exp(kappa).
Therefore:

* L>=0, with equality if and only if Q=y; the deterministic Bellman fixed
  point is unchanged.
* dL/dQ=eta g((Q-y)/eta)>-eta. Its derivative with respect to Q is at most
  exp(kappa): the scalar loss gradient is globally Lipschitz.
* Positive residuals (overprediction relative to the frozen target) receive
  a larger penalty than negative residuals of equal magnitude.
* This scalar curvature bound is NOT a bound on neural parameter gradients;
  the existing global gradient clip remains necessary as an implementation choice.

Kappa=4 bounds the exponent by exp(4), while residuals keep their gradients
above that threshold. The implementation never calls exp on an unclamped
positive residual, including the unused branch of autodiff.

A true uniform deterministic loss bound L(s,a)<=ell can be turned into a
residual bound. Since phi_k(x)>=phi(-|x|), let h be the inverse of
t -> exp(-t)+t-1 on t>=0. Then

    |Q-y| <= eta h(ell/eta^2).                              (7)

`residual_bound_from_uniform_loss` computes this inverse. Apply (5) if y uses
a delayed target. This helper cannot turn a sample mean or sample maximum into
a global certificate.

## 6. Stochastic targets: quantify the changed objective

For random Y conditioned on (s,a), the unrestricted regression minimizer is

    R_{eta,k}(Y) = argmin_q E[eta^2 phi_k((q-Y)/eta)],
    E[g((R_{eta,k}(Y)-Y)/eta)] = 0.                         (8)

It generally differs from E[Y]. For the uncontinued exponential it equals
the entropic soft minimum R_e(Y)=-eta log E[exp(-Y/eta)].

For the continued loss,

    R_e(Y) <= R_{eta,k}(Y) <= E[Y].                         (9)

Proof: g(x)<=exp(x)-1, so the expected g at R_e is nonpositive and its root
is no smaller. Convexity of g and Jensen imply E[g((E[Y]-Y)/eta)]>=g(0)=0,
so its root is no larger than E[Y]. Strict monotonicity gives uniqueness.

If Y has support in an interval of width W, Hoeffding's lemma gives

    0 <= E[Y]-R_{eta,k}(Y) <= W^2/(8 eta).                 (10)

For deterministic full-state dynamics and rewards, W=0 and the bias vanishes.
MuJoCo being a simulator does not prove that observations contain the complete
Markov state; (10) should not be reported as zero without checking that premise.

R is monotone under an almost-sure ordering of random variables and translation
equivariant. Proof: compare their increasing score equations in (8), and
substitute q+c and Y+c. Consequently R is 1-Lipschitz in the coupling sup norm.
The fitted operator

    T_{eta,k} Q(s,a) = R_{eta,k}(r+gamma m F(Q)(s') | s,a)

is therefore a gamma contraction on bounded Q functions. Its fixed point
Q_{eta,k}* exists uniquely, and if a uniform W bound applies at that fixed point,

    ||Q_{eta,k}*-Q*|| <= W^2 / [8 eta (1-gamma)].           (11)

More generally, if ||Q-T_{eta,k}Q||<=epsilon_R and W is a valid uniform
conditional target-width bound for this Q, use
epsilon=epsilon_R+W^2/(8 eta) in (1)-(3). This separates regression error from
stochastic-target bias. Neither a neural optimizer nor an empirical replay
loss is claimed to attain epsilon_R. Do not call the minimum of two heads
a confidence bound; this revision does not add such heads.

## 7. Validation and experiment scope

`validate_energy_bellman.py` checks the implemented loss against generalized
KL, its derivatives and numerical tails, finite-MDP contraction/residual/
policy bounds (including termination), stochastic bias and deterministic
limits, and the real trainer's gradient/EMA path on a small contextual task.
Numerical checks are regression checks for the proofs, not proofs themselves.

The first MuJoCo pilot changes only `--loss-kind relative_energy` with eta=10
and kappa=4. Baseline: frozen Huber run, same environment/seed and 100k prefix.
Metric: fixed-seed native stochastic-policy mean return, TD error, Q/V scale,
and tail fraction. Hypothesis: retaining the energy-divergence geometry helps
prevent persistent optimistic errors. No performance improvement is assumed.
All experiment tracking remains local/offline.

## Sources and distinction from prior algorithms

Exa Search, 2026-09-22: six search queries, 42 returned result slots, with
duplicate papers and irrelevant results filtered. Primary material inspected:

1. [Haarnoja et al., Soft Q-learning (2017)](https://proceedings.mlr.press/v70/haarnoja17a.html):
   Gibbs policies and the soft Bellman framework.
2. [Nachum et al., Path Consistency Learning (2017)](https://proceedings.neurips.cc/paper/2017/file/facf9f743b083008a894eee7baa16469-Paper.pdf):
   value/log-policy consistency and unified representations. Not implemented here;
   changing the trajectory objective at the same time would add another variable.
3. [Dai et al., SBEED (2018)](https://proceedings.mlr.press/v80/dai18c/dai18c.pdf):
   the distinction between Bellman-operator contraction and convergence with
   nonlinear approximation; stochastic double-sampling issues. No SBEED convergence
   guarantee is imported into this different optimizer.
4. [Garg et al., Extreme Q-learning (2023)](https://arxiv.org/abs/2301.02328):
   exponential/Gumbel regression. Its usual value-fitting residual is target minus
   prediction, whereas (6) uses prediction minus target derived from GMM forward
   generalized KL. This revision is not XQL and does not assume Gumbel TD errors.
5. [Omura et al., Stabilizing Extreme Q-learning (2024)](https://arxiv.org/abs/2406.04896):
   warns about exponential-loss instability and uses Maclaurin expansions. Our
   positive-tail quadratic continuation is a separate construction with (9)-(10).
6. [Chao et al., MEow (2024)](https://arxiv.org/abs/2405.13629):
   exact joint policy/value representations and practical value stabilization.

Equations (6)-(11), the continuation choice and its application to this spline
are our derivation/adaptation, not empirical findings or attributed claims of
the cited papers. No novelty priority claim is made.
