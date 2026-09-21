# Density correction warm start (Yonsei)

User requested density beta=0 initially, linearly restored to1 by100K.
This is a new comparison; existing Vast directMLL beta1 and DIPO stay untouched.
Same five environments x seeds0/1/2, T=.25, 1M steps, 64 conditional components
and64 exact-mixture candidates, direct marginal NLL, learned mu/sigma, no OT.

beta(t)=min(1,max(0,t/100000)), where t is ENVIRONMENT steps, not optimizer
steps and not steps since warmup. At the first post5K update beta~.05001;
at50K it is.5; from100K onward it is1. Q/T is unchanged:
w=softmax(Q/.25 - beta(t)*log q_old). This is not tempering the entire logit.

No archived or already-running learner source is changed. A subclass supplies
the beta scalar to the unchanged jitted learner on each train call, restores
the terminal-beta config, and logs train/density_beta on every update.
The stored config records the schedule explicitly in experiment metadata.

Use Slurm only, one GPU/4CPU/24GB per job, big_qos (no preemption), no requeue,
24h maximum. Validate actual CUDA initialization and full B256 learner on a
GPU allocation BEFORE submitting the15-run array. Source must be committed
before launch. Use existing private netrc; never commit or echo passwords.
W&B: OptiQ/heejoon-direct-mll-beta-anneal.

Runtime reuses the existing immutable Yonsei MuJoCo2.3.7 overlay at
/scratch/manfromearth_11/optiq-resampled-cost50-20260920/venv.
Data/code/checkpoints under /scratch/manfromearth_11/optiq-mll-beta-20260920.
Node selection must follow fresh CUDA diagnostics, not only idle Slurm status.
