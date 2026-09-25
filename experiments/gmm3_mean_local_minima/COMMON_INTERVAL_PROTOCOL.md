# Common interval [5,6] pilot and distinction between bad minima and saddles

Keep the previous target means (-1.5,1.5,6), sigma .5, weights 1/3,
population NLL, unconstrained mean-only GD lr .01, and 100K updates.
Sample all three initial means independently from Uniform[5,6], seeds 0-3.
Also test (5,5+1e-8,6), (5,5+1e-12,6), (5,5,6), and (5.5,5.5,5.5).
These stress tests must be reported even if they contradict the hypothesis.

This is an exploratory interval candidate; do not call four observed
failures a proof for every initial point, or a probability-one statement.

There is a simple exact-arithmetic obstruction to the literal universal
claim for a common interval I: I^3 contains diagonal initializations.
If mu1=mu2=mu3=m then all responsibilities are 1/3 and

    m_next = m - eta (m-E[X])/(3 sigma^2).

For 0<eta<6 sigma^2, this converges on the invariant diagonal to E[X].
The NLL Hessian there has eigenvalue 1/(3 sigma^2) along the diagonal,
and (sigma^2-Var[X])/(3 sigma^4) in each of the two contrast directions.
For any nondegenerate equal-weight, common-sigma target GMM,
Var[X]=sigma^2+Var[target means]>sigma^2. Thus this point is a strict
saddle, not a bad local minimum. For our target E[X]=2 and Var[X]=9.75,
the eigenvalues are (-50.6666667,-50.6666667,1.3333333).
Floating-point arithmetic can break exact equalities; record the actual
endpoint separately from this analytic statement. Excluding the measure-
zero diagonal does not by itself prove almost-sure bad-minimum convergence.

An alternative is a joint parameter region around a strict bad local
minimum. Continuity of the Hessian ensures a sufficiently small ball with
0<m I <= Hessian <= L I. For eta<2/L, GD contracts toward the minimum
and stays in that ball. This proves existence of an attracting region,
but a numerical radius needs separate bounds. A component-specific box
inside that ball can implement independent initializations with guaranteed
bad-minimum convergence; it is not the same as using a shared interval.

Source/protocol committed before launch. Same initial-value pilot runner
as range_probe, with an explicit config argument. Keep prior snapshots
immutable. Use CPU resources, not the active GPU jobs.
