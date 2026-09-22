# AntMaze v3 OptiQ: NovelD coefficient, 100k steps

User request: use server 180's four idle GPUs for NovelD coefficients
0.1, 1, 5, 10, stop each at 100,000 environment interactions and inspect results.
The preceding measured example was v3 OptiQ; use that task and training seed 0.
Each is a fresh run from the same initialization, not a continuation of a policy
already concentrated on one route. Exactly four production runs, one per GPU.

Only the NovelD coefficient changes. Keep dense reward=-nearest-goal distance,
novelty difference discount .5, normalize=false, RND AdamW1e-4/clip1 and one RND
update per learner update. Replay stores raw environment reward and adds the
recomputed intrinsic bonus when sampled. NovelD default remains .01 globally.

Keep OptiQ T=.25, beta=1, DACER enabled/target -.9 per action dimension,
noise_scale .1, initial_alpha .27, alpha_lr .03, interval10k, GMM3/200samples;
256x2 GELU, mean-init1, logstd[-5,-1]/initial-1, random z, N=M64, batch256,
UTD1, actor/critic Adam3e-4, gamma.99, tau.005, warmup5k, replay1M, one env.
Each production run therefore performs 95,000 learner and NovelD updates.

W&B: OptiQ/gmm-trg, separate group
antmaze-v3-optiq-noveld-strength-100k-s0-20260922.
The previous 16-run campaign and its frozen19fc37a source are unchanged.
Commit source/protocol/controller before preflight or production.

New-setting preflight: verify multiplier linearity on identical observations,
unchanged predictor/target updates across coefficients, and that Replay actually
passes the augmented reward to the learner. Then each coefficient runs a small
272-step smoke (16 updates) and a new-process resume to280 (24 updates), using
the same coefficient. Do not rerun the old 16-run preflight or physics suite.
All four must pass before production. Compare initial parameter/RND hashes and
all non-ablation config fields across the four production jobs.

Use supervisor, GPU wrapper locks, no auto-restart. Failure holds pending work
and preserves live work. No extra seeds, coefficient values, or extensions.

Evaluate at0/25k/50k/75k/100k: native(random-z mu-only) and direct policy
(random-z plus conditional sigma), 10 episodes each. Final100 episodes per mode
and natural/fixed-full-state condition, plus OptiQ zero-z. No extra DACER noise
or NovelD reward at evaluation. v3's natural start already is fixed, so never
pool natural and fixed into 200 independent initial states. Label sigma modes.

Save full final100k replay/model/target/optimizer/entropy/DACER/NovelD/RNG/env
checkpoint with SHA256. Archive locally and validate every replay reward and
all final raw rollouts. Report each seed0 policy separately: success/failure,
goal/route fractions, first training success by goal, goal distances, spatial
coverage and learning curves. 100k is an early exploration probe, not proof
that a setting failing by100k cannot learn later. No selecting a winner by
coverage alone while hiding goal failures.
