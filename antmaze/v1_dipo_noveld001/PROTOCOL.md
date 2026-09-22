# v1 DIPO control

User selection on 2026-09-22: retain NovelD 0.01 on v1 and add DIPO only.
Other coefficient studies are restricted to v2/v3/v4, with candidates not yet selected.
Do not register or resume the previous all-maze coefficient-10 queue.

One fresh v1 DIPO seed 0 run, 1,000,000 environment interactions; existing
OptiQ/SAC/MFPO v1 coefficient-0.01 results are preserved as controls.
The DDiffPG map, dynamics, nearest-goal dense reward and existing NovelD implementation
remain unchanged. No xy-only novelty, episodic first-visit mask or new shaping.

NovelD coefficient .01; novelty_discount .5; no normalization. Native DIPO
100-step diffusion, 20 action-gradient steps, 256x3 Mish, actor/critic Adam
3e-4 (eps 1e-5), action LR .03, batch 256, UTD 1, warmup 5000, replay 1M.
The pre-existing corrected persistent best-action replay is retained.
Shared replay computes NovelD once per critic minibatch and stores only dense
environment rewards; terminal masks are converted to gamma*(1-terminal) for DIPO.

Evaluate native and direct-policy sampling every 250k, plus initial and final
evaluations. Save the full continuation checkpoint only at final 1M.
Native DIPO retains random initial diffusion noise with reverse-step noise off;
direct policy retains the native stochastic reverse diffusion. Neither adds
external DACER noise or intrinsic reward to evaluation. Final natural/fixed
full-state rollouts: 100 episodes per evaluation mode/reset condition.

Before production, a committed-source 272-step smoke run with 16 learner updates
is verified and restored in a new process through 280 steps. Validate complete
model/target/optimizer/best-action memory/RND/replay/RNG/simulator state, actual
replay dense rewards and raw trajectory metrics. Smoke artifacts are not results.

Controller uses one free GPU under existing locks and supervisor, no automatic
restart and no follow-up jobs. Frozen source and prior checkpoints are preserved.
Archive the final state, replay, evaluations and provenance locally after completion.
