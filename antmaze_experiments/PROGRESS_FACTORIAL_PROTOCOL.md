# OptiQ AntMaze distance progress x goal bonus factorial

User approval 2026-09-24: fresh OptiQ only, four reward conditions x v1-v4,
seed0 each, total16 policies. This supersedes the implementation-only hold for
these profiles. No other experiment or cancelled job is restarted/stopped.

| Profile | Distance | Terminal success bonus |
| --- | --- | --- |
| progress_euclidean | nearest goal Euclidean XY | upstream10/20 |
| progress_euclidean_no_bonus | nearest goal Euclidean XY | zero |
| progress_geodesic | nearest goal obstacle-avoiding XY | upstream10/20 |
| progress_geodesic_no_bonus | nearest goal obstacle-avoiding XY | zero |

All profiles: r=d(current)-d(next)-0.01+bonus. There is no gamma inside this
formula. Bonus ON retains v2(-8,8)=20, v2(8,0)=10, and10 for all other goals.
Bonus OFF removes the reward only: goal radius0.5, success reporting and immediate
termination remain exactly the same. Actual terminal distance is not reset to0.
Timeouts retain bootstrap. NovelD OFF in all16. See PROGRESS_REWARD_PROTOCOL.md
for continuous visibility-graph geometry, numerical margin1e-6m, goal coordinates
and physical limitations (point torso XY, not Ant joint-space planning).

Use the existing DACER-OFF T1 control settings, source484f92e7d6d34c964d85b4493ff17c5a9ebcf32e:
constant T1, DACER OFF, beta1, random Gaussian latent, conditional sigma sampled
in training/direct-policy evaluation; mu-only native evaluation. Actor and twin
critics256x3 GELU, mean-init1, N=M64, log sigma[-5,-1]/initial-1, actorLR3e-4,
criticLR5e-4, Adam without clipping, gamma.99, tau.005, plain TD. Replay1M,
256CPU envs, batch4096,8updates/256transitions,8192warmup. No reward normalization,
extra exploration, architecture or critic changes for OptiQ.

Original native post-warmup budgets: v1/v2 3M, v3 4M, v4 5M. Strict-greater
counter and vector rounding yield total3,008,256/3,008,256/4,008,448/5,008,384.
Each job independently passes8448transition/8update preflight with full replay
reward verification, finite parameter updates and policy save/restore before
fresh main training. Commit/push/share frozen source before preflight and launch.

Host180 runs v4 four conditions first, then v1 four; host199 v3 then v2.
Existing GPU locks are respected. Fill any free local slot every2seconds with
the next job; no cross-maze/condition/preflight barrier. Existing running work
continues. Failure holds pending jobs and preserves other live jobs. No automatic
restart. Supervisor controller and W&B synchronization service manage lifetime.
Project OptiQ/antmaze; group antmaze-optiq-progress-2x2-T1-s0-20260924.

Evaluate every250k using40 random-start episodes per native/direct mode; save
evaluation-only policies at those checkpoints. Final100episodes per native/
direct/zero-z mode and random/fixed starts, plus full model/optimizer/replay/RNG/
simulator state. Primary plots use random-start direct-policy rollouts; mu-only
and fixed-start supplementary. Keep original training resets and evaluation-only
xy uniform[-2,2]. All16 use the same seed/evaluation-start protocol.

Report success, corridor usage, route persistence, visitation area and failures;
do not merge seeds/policies or equate entropy with multiple successful routes.
Raw returns differ between these reward definitions: compare common success/
trajectory/distance metrics rather than ranking raw return across conditions.
Each condition has just one seed. Full replay verification uses the matching
reward profile including terminal bonus OFF; historical dense-only collectors
must not be used for these profiles. Preserve all frozen sources and old results.
