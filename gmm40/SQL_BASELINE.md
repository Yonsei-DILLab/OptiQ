# JAX SVGD Soft Q-Learning baseline

Source: <https://github.com/haarnoja/softqlearning.git>

Pinned commit: `6f51eaca77d15b35c6443363c51a5a53ff4e9854`.
The unmodified upstream repository remains the `gmm40-baseline/SQL` Git
submodule; [baselines.json](baselines.json) records the same pin. The runnable
JAX/Flax/Optax port is [sql_jax.py](sql_jax.py), with GMM40 adapters in
[sql.py](sql.py). Both fixed-Q reconstruction and online navigation select it
with `--method sql`. The port does not import TensorFlow or garage.

```bash
git submodule update --init --recursive
```

## Source correspondence

Paths below are relative to `gmm40-baseline/SQL`:

- `softqlearning/algorithms/sql.py`: `SQL._create_svgd_update` samples policy
  particles, splits fixed and updated particles, forms the score from Q plus
  the squash correction, and combines kernel-weighted score attraction with
  the kernel-gradient repulsion. It propagates the resulting action gradient
  through the policy network using Adam. `_create_td_update` estimates the
  soft next-state value using uniform action samples and log-sum-exp.
- `softqlearning/misc/kernel.py`: adaptive isotropic Gaussian (RBF) kernel;
  bandwidth is median squared distance divided by log(number of fixed
  particles), with a minimum of 1e-3.
- `softqlearning/policies/stochastic_policy.py`: stochastic neural policy.
- `examples/mujoco_all_sql.py`: original MuJoCo training entry point.
- `examples/multigoal_sql.py`: original multi-goal example.

The JAX port preserves separate first-layer projections for observations and
noise/actions, Glorot-uniform weights, zero biases, ReLU hidden layers and
tanh policy outputs. Fresh standard-normal noise has dimension `action_dim`.
It retains a single scalar Q and hard target copies; there is no twin-Q,
Polyak update, learned entropy coefficient or gradient clipping.

Important numerical details retained from the pinned implementation:

- The score is differentiated with respect to bounded actions, from
  `Q/T + sum(log(1 - a*a + 1e-6))`. Both the kernel and its repulsive gradient
  operate on those bounded actions. This deliberately preserves the upstream
  coordinate convention.
- The kernel's bandwidth uses the **lower middle value** when the number of
  distances is even, matching upstream `top_k`; an averaged median differs.
- The fixed-particle axis is averaged; the batch and updated-particle axes
  are summed in the policy parameter gradient. No extra batch reduction is
  inserted. The SVGD direction is stopped before the policy VJP.
- Adam uses epsilon on the uncorrected second moment, as in TensorFlow 1.x.
- The soft backup uses one uniform action cloud shared across states and the
  `action_dim * log(2)` volume correction. Terminal transitions do not bootstrap.
- Actor and critic gradients read the pre-update live Q. Target Q is copied
  after updates when the supplied zero-based environment iteration is divisible
  by the interval. Initially target Q equals live Q. `SQLLearner.update` uses
  learner steps when no explicit environment iteration is supplied.

## Experiment settings

| Setting | JAX GMM40 default | Upstream MuJoCo example |
|---|---|---|
| Policy / Q hidden layers | 256 x 2 | 128 x 2 |
| Batch | 256 | 128 |
| Actor / critic LR | 3e-4 | 3e-4 |
| Training updates per environment step | 1 | 1 |
| Kernel particles / updated fraction | 16 / 0.5 (8 fixed x 8 updated) | 16 / 0.5 |
| Uniform value particles | 16 | 16 |
| Hard target interval | 1000 environment steps | 1000 environment steps |
| Discount | 0.99 | 0.99 |
| Reward scale | 1 (navigation) | Environment-specific |
| Temperature | 1 | Implicit 1, controlled through reward scale |

The explicit temperature option is an extension: the policy uses `Q/T` and
the learned soft backup uses `T*log integral exp(Q/T) da`. At T=1 these are the
original equations. Fixed-Q uses only `log p_GMM(40*a)` energy queries, without
target samples or a learned critic. Evaluation uses fresh policy noise and an
isolated RNG. OptiQ-specific `--n`, `--m`, OT and log-sigma settings do not apply
to SQL. The run config records the SQL settings and source hashes.

## Running and checking

Use the existing GMM40 Python 3.11 / JAX environment. No legacy TF1 installation
is needed for this port; the original submodule remains available for reference.

```bash
# Fixed-Q density reconstruction. 64 x 64 SVGD pairs would use 128 particles.
python -m gmm40.run --method sql --name sql_fixed_seed0 --seed 0 \
  --steps 100000 --batch 256 --width 256 --depth 2 --temperature 1

# Online GMM40 navigation: 5K warmup, then one update per environment step.
python -m gmm40.run --method sql --navigation --name sql_navigation_seed0 \
  --seed 0 --steps 100000 --warmup 5000 --temperature 1

# Independent numerical and checkpoint/RNG checks; no training campaign.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -m gmm40.validate_sql
```

Particle controls are `--sql-kernel-particles`, `--sql-kernel-update-ratio`,
`--sql-value-particles` and `--sql-target-update-interval`. Network size and batch
use the common CLI flags. Fixed-Q `--resume` verifies the learner configuration,
batch and target identity. Online checkpoints include both optimizers, target Q,
training RNG and a replay/RNG sidecar. `SQLOnline.restore` restores these;
navigation CLI resume is rejected because environment restoration is not wired.

`SQLLearner` supports arbitrary flat state/action dimensions; `SQLOnline` supports
flat Gymnasium spaces with actions normalized to [-1,1]. The integrated CLI
environments here are fixed-Q GMM40 and GMM40 navigation. MuJoCo campaigns have
not been launched. Commit the exact experiment settings before launching long
runs, and use supervisor on managed instances.

Validation compares kernel values/gradients, score, policy parameter gradients,
Adam steps and soft TD/critic gradients against independent NumPy/PyTorch
calculations of the pinned equations. It also checks target cadence, deterministic
checkpoint continuation, replay wraparound, and evaluation RNG isolation. The
legacy TensorFlow runtime is not executed, and no cross-framework bitwise
training-trajectory equivalence or benchmark performance is claimed.
