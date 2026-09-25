# N=M256 memory-compatible follow-up, 2026-09-25

The initial N=M256 RTX5090 preflights using source
`b0df33807a05debd3df47536c34058b0962d6c8f` failed at the first actor
update with `RESOURCE_EXHAUSTED`: XLA requested about 8.5 GiB after filling
GPU memory. Their logs, failure JSON, manifests and frozen source remain.
No N=M256 main learner started from that source.

This new frozen source computes the direct GMM actor loss and gradient over
16 independent 256-state chunks of the same 4,096-state replay minibatch,
averages those gradients and calls Adam **once**. Critic training, the actor
objective, N=M256 candidate structure, effective batch4096, replay draws,
learner-update count, warmup, reward, all model settings and evaluation are
unchanged from `UTD256_NM_PROTOCOL.md`. Each chunk has its own PRNG subkey;
the stochastic draw is distribution-equivalent but not bitwise identical to
the single-array realization. Diagnostic means average over chunks and
minimum/maximum statistics reduce across them. N=M128 jobs continue on their original
frozen source; T=3 controls also continue.

Before main learning, each N=M256 run must pass 256 real batch4096 learner
updates, including full source/config/gradient-counter/checkpoint checks.
If memory or numerical checks fail again, hold the jobs and preserve the
failure. Never silently reduce N/M, the effective batch, UTD, or reward.
