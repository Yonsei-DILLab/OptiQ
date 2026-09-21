# OptiQ v5 GMM40 experiments (imported into direct-gmm-trg)

Fixed-Q density reconstruction and a separate 100-step navigation task using
the DiKL seed-0, 40-component Gaussian mixture. The default OptiQ adapter reuses
the v5 actor, Gaussian proposal, mean-action Sinkhorn assignment and full-row
conditional Gaussian NLL. Baselines include SAC, DIPO, MEow and MFPO.

## Source and algorithm

Imported from `v5-gmm40` commit `a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71`.
[IMPORT_SOURCE.json](IMPORT_SOURCE.json) records the files and dependency trees.
The historical v5 implementation and Hydra profiles are pinned under `_v5/`;
its Python namespace is `gmm40._v5.optiq_dime`. Shared `common`, `models` and
`diffusion` dependencies were checked byte-for-byte against the source branch.

`--method optiq` uses the original mean-action OT/Sinkhorn adapter, including
its historical actor sigma settings. It does not select the Direct GMM/TRG
trainer. Active TRG defaults and running frozen snapshots are separate.

## Dependencies

Use the repository's Python 3.11 / JAX environment (`requirements-mujoco.in`)
with `gym==0.26.2` for the upstream MFPO adapter. The recorded environment used
JAX/JAXlib 0.4.33, Flax 0.9.0, Optax 0.1.7, NumPy 1.26.4 and SciPy 1.11.4.
The PyTorch agents require a CUDA-enabled PyTorch build; the experiment used
PyTorch 2.4.1+cu124, whereas the MuJoCo requirements specify CPU PyTorch.

The four original baseline repositories are Git submodules, pinned to the
revisions in [baselines.json](baselines.json): DiKL, DIPO, MEOW and MFPO.
After checking out this branch, initialize their source code with:

```bash
git submodule update --init --recursive
```

Their paths are `gmm40-baseline/DiKL`, `gmm40-baseline/DIPO`,
`gmm40-baseline/meow`, and `gmm40-baseline/MFPO`. The upstream files are
unmodified and retain their upstream licenses. DiKL initializes the target,
including for OptiQ-only runs. Generated results are excluded from Git.

## Running

```bash
# Fixed Q = log p_GMM, T=1: 100K actor updates, one selected GPU.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m gmm40.run --method optiq --name optiq_seed0_100k \
  --seed 0 --steps 100000 --n 16 --m 64 --temperature 1

# Navigation: 5K random warmup followed by 100K actor updates.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m gmm40.run --method optiq --navigation --name navigation_optiq_seed0_100k \
  --seed 0 --steps 100000 --warmup 5000 --temperature .25
```

Use a fresh name for every run. On managed instances, launch long runs through
supervisor. `python -m gmm40.run --help` lists explicit experimental overrides,
including N/M, Sinkhorn iterations, NLL row selection and sigma row balancing.
These options are disabled by default. Fixed-Q T other than 1 changes the
target to a tempered density; navigation learns its critic separately.

Results, target metadata, source hashes, checkpoints, stochastic-policy samples
and evaluations are written under `gmm40-results/`. Reference target samples
are used for evaluation only. Physical actions are scaled by 40; the primary
density target is conditioned on the box (-40, 40)^2.

`dispatch.py` is the original instance-specific campaign launcher. It expects
a local `gmm40-results/queue.json`, `/root` virtual environments and CPU affinity
assignments; use `gmm40.run` directly on other machines. Historical diagnostic
and comparison scripts require the named local checkpoints/results in their
source. Those large artifacts are not distributed in Git.

## Checks

```bash
# CPU-only target, gradient, MEOW Q/V, v5 NLL and navigation invariants.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -m gmm40.validate

# CLI/configuration discovery (does not train).
python -m gmm40.run --help
```

The historical diagnostic and comparison scripts still require their named
local checkpoints. Model archives and generated results are not included.
