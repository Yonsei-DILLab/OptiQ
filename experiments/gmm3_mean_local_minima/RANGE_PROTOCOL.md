# Does initialization in [0,6]^3 force a bad local optimum?

Keep the previous target (-1.5,1.5,6), equal weights, sigma 0.5, float64
population NLL and mean-only GD unchanged. Only the initial means must
lie in [0,6]. Subsequent GD updates are UNCONSTRAINED; projecting parameters
to [0,6] would make the negative target mean unrepresentable and answer a
different question.

Before running, select four explicit starting points listed in
range_probe.json, plus four iid Uniform[0,6] initializations (seeds 0-3).
Use 100K updates, learning rate 0.01, the same 4097-point quadrature.
Record initial/final means, initial gradient, trajectory, endpoint loss
gap and gradient norm. Independently verify endpoints with adaptive
quadrature.

One successful initialization is sufficient to refute the universal
claim. Four random seeds are illustrative and insufficient to establish
a reliable failure probability. No attempt is made to prove a basin
boundary or a universal convergence theorem. All eight runs are reported.

Commit before running. Record commit and Slurm job in the manifest and
result. Reuse the small CPU Slurm resources, without affecting GPU jobs.
