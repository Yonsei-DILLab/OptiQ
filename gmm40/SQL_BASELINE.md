# Original SVGD Soft Q-Learning baseline

Source: <https://github.com/haarnoja/softqlearning.git>

Pinned commit: `6f51eaca77d15b35c6443363c51a5a53ff4e9854`.
The unmodified upstream repository is the `gmm40-baseline/SQL` Git submodule;
[baselines.json](baselines.json) records the same pin. This is the ICML 2017
Soft Q-Learning implementation, including its amortized SVGD policy update.

```bash
git submodule update --init --recursive
```

## Implementation

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

The MuJoCo example defaults to policy/Q networks of 128 x 2, batch 128,
policy/Q learning rates 3e-4, one training repeat, 16 kernel particles with
update ratio 0.5, 16 value samples and target update interval 1000.
Reward scale, environment version and training duration vary by environment.
These are upstream defaults, not the current OptiQ 256 x 2 protocol.

## Runtime and integration status

This addition distributes and pins the original source. There is no SQL
adapter in `gmm40.run`, no port to the current JAX/PyTorch trainer, and no
SQL training has been launched or validated by this import.

Upstream `environment.yml` requests Python 3.6.6, TensorFlow >=1.9,<1.10,
NumPy 1.14.5, Gym 0.10.8 and mujoco-py >=1.50.1,<1.50.2. The README describes
an old rllab checkout and MuJoCo 1.31, while this commit's actual imports use
`garage`. The historical instructions and dependencies therefore need
compatibility work before execution; they are not a verified installation
recipe for the current servers. Keep any future legacy runtime isolated from
the active OptiQ virtual environments, and define a matched experiment
configuration before making performance comparisons.
