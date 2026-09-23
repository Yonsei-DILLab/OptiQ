# OptiQ AntMaze: dense reward, NovelD OFF, temperature 1

User approval on 2026-09-24: stop running SAC jobs and start four fresh OptiQ
policies, one seed0 policy for each v1-v4 maze, with dense reward, NovelD OFF
and temperature1. Preserve completed runs and running DIPO jobs. Do not restart
held MFPO or cancelled SAC/older campaigns as part of this experiment.

## Controlled change

Relative to the OptiQ jobs in `antmaze-dense-off-16-current-s0-20260924`
(training source `a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3`), change only the
learning temperature from0.01 to1.0. Use the current dedicated
`direct-gmm-trg-antmaze` branch; dependency checks added after the reference
campaign do not alter OptiQ's learning algorithm. Do not change shared defaults.

- Actor and twin critics:256x3 GELU; actor LR3e-4, critic LR5e-4, Adam, tau.005.
- Beta1, DACER enabled, behavior noise scale.1; mean initialization1,
  random latent,N=M64,log sigma[-5,-1],initial-1.
-256 CPU Gym environments, batch4096,replay1M,warmup8192,
  eight learner updates per256 transitions (UTD1/32).
- Native budgets including warmup and strict-greater-than rounding:
  v1/v2 3,008,256;v3 4,008,448;v4 5,008,384 transitions.
- Evaluation every250k with40 episodes per native/direct mode, random xy
  starts uniformly in[-2,2]. Preserve original training reset distributions.
  Save OptiQ evaluation policy snapshots every250k; final100 episodes per
  mode/reset and final full replay/model/optimizer/RNG/simulator checkpoint.
- Primary direct-policy evaluation includes random z and conditional sigma;
  native random-z mu-only and fixed-state evaluation remain separate. No
  external DACER behavior noise or intrinsic reward in evaluation.

## Dense reward audit

MaxEntDP Appendix D.2 states that its AntMaze reward penalizes distance from
the closest goal: https://arxiv.org/html/2502.11612v3#A4.SS2 . Our external
`Recorded.step` wrapper implements `-min_g ||next_xy-g||_2`, replacing (not
adding to) the original sparse10/20 reward. No squared/geodesic distance,
progress bonus, success bonus, action cost, normalization or scaling is added.
NovelD is off: no intrinsic bonus, RND model, optimizer or RND updates.

This agrees with the paper's stated reward principle. The appendix does not
specify an exact formula/coefficient, pre/post-transition convention or full
termination implementation, so it does not establish an exact reproduction of
all MaxEntDP environment details. Our environment preserves DDiffPG physics,
goal radius and termination. Goal detection terminates at radius0.5, so the
terminal dense reward can remain slightly negative. Auto-reset observations
are replaced by terminal observations before replay storage. Preflight/final
checkpoint checks compare every stored replay reward against next-state goal
distances. Upstream vendored files and the reward implementation are unchanged.

## Launch and provenance

Campaign `antmaze-optiq-dense-off-T1-s0-20260924`; W&B `OptiQ/antmaze` with
that campaign group. Server180 runs v1/v3, server199 v2/v4, respecting GPU
locks and existing DIPO jobs. Each job independently passes the actual
256-env,batch4096,eight-update preflight, checkpoint/replay verification and
temperature/architecture/LR assertions before full training starts. Commit,
push and share the exact frozen source first. Failure holds pending jobs;
no automatic restart or performance-based early stopping.

SAC cancellation is recorded in the old campaign's sidecar; the old controller
may label requested termination as failure. Preserve that raw status and logs,
and distinguish user cancellation from a new numerical/training error.
