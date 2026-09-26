# PointMaze DIPO upstream-memory replacement, seeds 0–2 (2026-09-26)

Latest user approval replaces only the running Medium/Hard DIPO seeds 1–3
with six fresh Medium/Hard seeds 0–2. Preserve all cancelled partial data and
old frozen sources. Existing seed-0, Simple, Way and other baseline runs remain
unchanged. This supersedes the DIPO part of POINTMAZE_MULTISEED_PROTOCOL.md.

Remove the vector adapter's DiffusionMemory.replace override and inherit the
pinned BellmanTimeHut/DIPO c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc method
literally. Upstream uses np.copyto on advanced-indexed actions, so it does not
persist improved actions across minibatches. Action-gradient improvement within
the current minibatch still runs for 20 steps and trains the diffusion actor.
This is fidelity to observed upstream behavior, not a claim that this behavior
is the algorithm's intended design or that replay writeback caused collapse.
Upstream files remain unchanged. Config records method identity and behavior;
preflight/final verification rejects the old writeback implementation.

For time, the user explicitly selected 32 learner updates per 2048 transitions
(UTD=1/64), reduced from 128 (UTD=1/16). Batch remains 4096. Thus two things
change relative to the previous campaign: replay writeback behavior and UTD.
Do not attribute a performance difference solely to either change.

Each run: original DrAC Medium/Hard sparse +100 task, seed 0/1/2, 2048 vector
environments, 8192 warmup transitions, 1,001,472 total transitions, 15,520
learner updates, native DIPO diffusion100/cosine, 20 action-gradient steps,
action LR .03, action grad clip .2, actor/critic Adam3e-4 eps1e-5, actor/critic
grad clip2, tau.005, gamma.99, replay1M. Evaluate every200k with200 episodes,
final500 episodes including obstacle/removal metrics, save checkpoints and
raw trajectories and automatically render them. Evaluation retains fresh
initial diffusion Gaussian noise and disables reverse-process noise, as in
official DIPO eval. No OptiQ-style conditional-sigma modification applies.

Plan: POINTMAZE_DIPO_UPSTREAM_U32_PLAN.json. Campaign and root basename:
pointmaze-dipo-upstream-u32-s0to2-1m-20260926. Host46 GPU2/4/5/6 run Medium/Hard
seeds0/1; host199 GPU0/1 run Medium/Hard seed2. Temporarily stop only the waiting
baseline guards on these slots before stopping old DIPO, then restore them
after replacement workers acquire the existing GPU locks. Preserve other jobs.
Each run independently passes 10,240-transition/32-update preflight before
fresh main training. Failure pauses its host's pending queue; no automatic
retries. Commit, push pointmaze and share exact immutable source before launch.

Collect config, progress, proofs, raw evaluations, checkpoints and source
manifests with SHA-256. Report new three-seed results separately from historical
writeback/128-update results; do not combine them into one seed aggregate.
