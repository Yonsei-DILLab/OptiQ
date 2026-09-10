# What v2's soft policy-improvement theorem does and does not establish

The requested standard is the idealized soft-policy-iteration guarantee used
in SAC, together with empirical stability and performance validation. It is not
a promise that every neural-network update increases real Humanoid return.
Exact policy evaluation alone is insufficient: the policy-improvement step must
also satisfy the condition below. Merely adding entropy to a TD backup does not
prove that an OT or likelihood projection improves the policy.

## Exact operator

Fix temperature T>0 and discount 0<=gamma<1. Assume the soft Bellman fixed points
exist and the value differences are bounded (bounded rewards and entropies are
one sufficient setting). Let Q_old be the exact soft Q of the old policy and

    F_Q(pi,s) = E_{a~pi(.|s)} Q(s,a) + T H(pi(.|s)).

For each state, the normalized Boltzmann policy is

    p_Q(a|s) = exp(Q(s,a)/T) / Z_Q(s).

With finite normalizer and well-defined entropy/KL,

    F_Q(pi,s) = T log Z_Q(s) - T KL(pi(.|s) || p_Q(.|s)).

Consequently exact Boltzmann extraction, pi_new=p_Q_old, gives

    delta(s) = F_Q_old(pi_new,s) - F_Q_old(pi_old,s)
             = T KL(pi_old(.|s) || p_Q_old(.|s)) >= 0.

Let T_new denote the new policy's soft Bellman operator. Then
T_new V_old - V_old = delta >= 0. Monotonicity gives
T_new^k V_old >= V_old; contraction yields V_new >= V_old and Q_new >= Q_old.
This improves the soft objective. It does not alone establish an increase in
unregularized benchmark return, which is tested separately.

## Where OptiQ's sampling and transport enter

Conditional on a fixed explicit proposal q with support covering the target,
fresh candidates weighted by exp(Q/T)/q consistently estimate the Boltzmann
target as sample count increases, under the usual finite-integral conditions.
This is why v2 retains beta=1, a matching joint proposal density, no hard support
cutoff and a common temperature for extraction and the soft backup.

An exact transport to that target, followed by exact distribution-preserving
distillation, realizes the exact operator above without derivatives through Q.
Finite candidates, finite Sinkhorn iterations, restricted conditional Gaussians
and neural optimization introduce approximation errors. The implemented full-OT
conditional NLL is an approximation method; its loss is not reverse KL to p_Q.
Neither a lower NLL nor a lower Wasserstein distance certifies improvement.

This distinction is stronger than a casual analogy to SAC: exact reverse-KL
projection with the old policy feasible can retain or improve F at each state.
Wasserstein or forward-likelihood projection within a restricted family does
not have that property. With shared neural parameters and stochastic optimization,
the practical implementations of both methods also leave the exact-operator
setting. These qualifications must accompany any theorem claim.

## Arbitrary OT candidates with an acceptance condition

An alternative ideal operator proposes any OT-distilled candidate and accepts
it only if delta(s)>=0 at every relevant state; otherwise it keeps the old
policy. The same proof applies, regardless of how that candidate was generated.
This requires exact statewise evaluation, not a positive replay-average score.

For an approximate critic with a verified uniform error epsilon_Q, valid new
entropy lower bounds L_new, valid old entropy upper bounds U_old, and a verified
sampling-error bound epsilon_MC, a sufficient statewise acceptance margin is

    E_new Q_hat - E_old Q_hat + T (L_new - U_old)
        - 2 epsilon_Q - epsilon_MC >= 0.

The IDAC mixture entropy lower bound and independent-mixture upper bound hold
in expectation. Individual samples are not bounds. Twin-critic disagreement is
not epsilon_Q. The implemented sampled guard uses a replay average and estimated
standard error; it is an empirical surrogate, not this exact statewise test.
Its optional two-standard-error safety margin is not required by the ideal
theorem and must be evaluated for its effect on learning speed and return.

## Approximation error

More generally, if delta(s)>=-epsilon uniformly, then

    V_new >= V_old - epsilon/(1-gamma).

For an exact Q and a candidate with a verified uniform reverse-KL projection
error KL(pi_new||p_Q)<=epsilon_KL, delta>=-T epsilon_KL and the bound becomes
T epsilon_KL/(1-gamma). Observed NLL, ESS or small W2 error is not epsilon_KL.
These are theoretical error bounds; they have not been numerically certified
for the Humanoid neural critics or actors.

## Finite-M entropy in policy evaluation

The exact theorem uses the true marginal policy entropy, not merely convergence
of the implemented finite-M TD backup. For a frozen policy, let L_M(s) be the
expectation of the self-inclusive IDAC entropy estimator. In general
L_M(s)<=H(pi(.|s)); its bias is not removed by averaging more TD batches with
the same M. The lower-bound gap tends to zero as M grows under the estimator's
convergence and integrability assumptions.

If 0<=H(pi(.|s))-L_M(s)<=epsilon_H uniformly and the lower-entropy Bellman
equation is solved exactly, the discounted series gives

    0 <= V_true - V_lower <= T epsilon_H / (1-gamma),
    0 <= Q_true - Q_lower <= gamma T epsilon_H / (1-gamma).

The extra gamma in the Q bound follows because Q's entropy contribution starts
at the next state. Thus the critic-error budget in the sufficient acceptance
margin above must include finite-M entropy bias as well as critic approximation
error. A small observed entropy-bracket gap at sampled states is useful
diagnostically, but is not a verified uniform epsilon_H. The current M=16
implementation remains an approximation to true-entropy soft policy iteration;
exact policy evaluation in the theorem assumes this error is absent or
explicitly bounded. This derivation is our application of the discounted
Bellman equations, not an additional claim attributed to IDAC.

## Counterexample to a W2-only guarantee

Consider one state, zero reward, and three actions at coordinates 0, .01, 1.
The Boltzmann target is uniform. In a policy class containing only
old=(.5,.5,0) and candidate=(2/3,0,1/3), the candidate is much closer to the
target in squared W2 (.0000333 versus .3267167), but has lower entropy
(.63651 versus .69315). Thus exact W2 projection within that class decreases
soft value even when Q is exact. A finite-MDP test verifies this counterexample
and the approximate-improvement value bound.

References: [SAC soft policy iteration, Appendix B](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
and [IDAC](https://arxiv.org/html/2007.06159). The transport/distillation
qualifications and acceptance/error-bound analysis here are our application of
soft policy iteration, not a claim that IDAC proves the OptiQ projection step.
