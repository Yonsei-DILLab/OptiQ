# AntMaze dense temperature sweep — 2026-09-24

User request: OptiQ v1/v3/v4 × temperature 10/5/3 × seeds 0/1/2 = 27
fresh runs, retaining current AntMaze defaults. This file and manifest preparation
do not assert that a remote queue has been registered.

- Reward: negative nearest-goal Euclidean distance at the next position.
  No extra step penalty, sparse bonus or potential-based shaping. NovelD OFF.
- 256 vector environments, batch 4096, 8 updates per 256 transitions,
  replay 1M, warmup 8192. Native budgets: v1 3M, v3 4M, v4 5M,
  excluding warmup and retaining upstream strict-greater stopping.
- Actor and critic 256×3, Adam actor LR 3e-4 / critic LR 5e-4, tau .005.
  N=M=64, density beta=1, DACER ON, log std [-5,-1], initial -1.
- Evaluate every 250K: 40 episodes, final 100. Random evaluation starts,
  unchanged native training reset. Save intermediate evaluation policies and
  final full checkpoint. W&B OptiQ/antmaze, separate campaign group.
- Seeds propagate to environment, Python, NumPy, PyTorch, JAX learner config,
  and warmup RNG. Environment seed base is seed×256 to avoid overlapping
  vector environment seed sets. Common evaluation seeds remain unchanged.
- Preserve existing running/pending experiments. Never restart cancelled jobs.
  Remote installation and GPU ownership must be verified before registration.

Seed 0 behavior is unchanged by the seed plumbing. Upstream antmaze/ is untouched.

Deployment approved: our Vast hosts, behind existing work. Use vast2/3/4/5;
vast1 remains disk-constrained and its DIPO slot is preserved. Separate runtime
inherits existing Python3.11 packages but overrides Torch with CUDA12.8 support
and installs original Gym0.23.1 / MuJoCo2.1 / mujoco_py2.1.2.14. Existing runtime
packages and live code are not modified. Each GPU job must pass the real
256-env,4096-batch,8-update preflight before main training. Runtime failure
does not start training; job failure holds that host's pending jobs. No restart.
