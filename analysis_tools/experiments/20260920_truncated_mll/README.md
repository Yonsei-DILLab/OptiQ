# Bounded truncated Gaussian direct MLL

Same 15 environment/seed pairs as prior directMLL64 (five MuJoCo tasks, seeds
0,1,2, one million steps). T=.25, beta=1, no OT, no gradient clipping, normal
continuous latents,64 components/64 independently sampled candidates. No extra
action noise or uniform replacement. Data center beta annealing is unchanged.

Separate optiq_dime package copied from committed archived implementation;
shared framework/config modules reused read-only. Entry train.py puts this
package ahead of the archived source. No running source snapshot is modified.

Gaussian centers tanh(raw_mu), logstd clipped[-5,-1], initial=-1.
User confirmed tanh centers after discussing hard-clip gradient blocking.
Coordinate inverse-CDF sampling from the box-conditioned Gaussian for rollout,
plain TD, and teacher. No action tanh or clipped unconditioned Gaussian.
Numerical CDF endpoint protection at float32 epsilon; inverse-CDF output has a
support roundoff safeguard (float32 produced -1.00000012 before this fix).
Mixture log density includes coordinate
normalization; NLL differentiates both logstd and normalization. Teacher/weights
stopped. Eval stochastic-z/zero-z both use bounded Gaussian center (not truncated
expectation). No entropy backup; unsupported historical modes not exposed.

Preflight tests cover SciPy density, integration, sampling, gradients, frozen
teacher, actor bounds, and config/import identity. Short GPU smoke before launch.
Supervisor one run/GPU; DIPO preserved; prior logs/checkpoints retained.
W&B OptiQ/heejoon-truncated-mll.
