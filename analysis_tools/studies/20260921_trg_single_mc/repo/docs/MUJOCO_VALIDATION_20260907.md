# Initial validation on 2026-09-07 (UTC)

Historical report for commit `2968f8f`. The later reference-alignment changes
supersede its 5-seed/3M/5M defaults and evaluation RNG isolation. See
[reference setting audit](MUJOCO_REFERENCE_AUDIT.md) for the current protocol.

Prepared from `critic-dime` commit `8b8fee13b2cbc4d90183d7b803fce033eadf0a19`
on the `heechan` branch. Validation runs record the base SHA plus dirty state;
the full benchmark has not been launched.

## Environment

- Python 3.11.16 in `/workspace/.venv-optiq-mujoco`.
- JAX/JAXlib/CUDA plugin 0.4.33, Flax 0.9.0, Optax 0.1.7.
- Gymnasium 0.29.1, MuJoCo 2.3.7, SB3 2.1.0, PyTorch 2.4.1+cpu.
- NVIDIA RTX 3090, JAX CUDA backend confirmed by GPU computation and training.
- `uv pip check`: all 113 installed packages compatible.
- Repository `.env` authenticated successfully; default W&B entity `sae_project`.
  Secret values are not included in Git or this report.

## Training integration checks

Each run used the full 2048x2048 twin critic and 256x256x256 actor, batch 256,
UTD 2, the approved [-1600, 1600] critic support, proposal std 0.1, clip 0.15,
and anchors enabled. Only the validation budget/protocol was shortened:
12 environment steps, warmup 8, checkpoint interval 10, one evaluation episode,
log interval 4. Each completed 8 gradient updates and final evaluation.

| Environment | Sampling | Beta | W&B run |
| --- | --- | --- | --- |
| Ant-v4 | stratified | 0.1 | [s5xnf7w9](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/s5xnf7w9) |
| Ant-v4 | exact | 0.25 | [cin5lihg](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/cin5lihg) |
| Humanoid-v4 | stratified | 0.5 | [ox9k3n1r](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/ox9k3n1r) |
| Humanoid-v4 | exact | 1.0 | [697cvixw](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/697cvixw) |

The W&B API confirmed `finished`, the correct environment and beta, GPU backend,
gradient updates, metric history and one uploaded experiment artifact per run.
Artifacts contain actor/critic checkpoints and saved per-episode evaluations.
All four final checkpoints were reloaded into matching architectures, checked
for finite parameters, and used for action and categorical critic inference:
[checkpoint reload validation](https://wandb.ai/sae_project/optiq_dime_mujoco_v4/runs/lp2dxd86).

These runs validate execution and persistence, not convergence or sampler quality.
Headless simulation was tested; interactive/video rendering was not tested.

## Automated checks and review

- 17 initial pytest cases passed, including all 100 composed task configs,
  both samplers' distribution/density consistency, anchor/component allocation,
  both v4 environments and timeout bootstrap, and finite actor updates for
  every sampling x beta combination.
- One additional evaluation test passed: evaluation preserves training action
  RNG state and writes timestep/return/episode-length arrays. Total: 18 cases.
- Python syntax, Ruff fatal-error checks, shell syntax, and `git diff --check` passed.
- Dry-run task table contains exactly 100 distinct runs (2 environments x
  2 samplers x 5 betas x 5 seeds).
- Supervisor template paths exist; four foreground workers, autostart disabled.
- Source review confirmed no changes to `models/critic.py`, `diffusion/dime.py`
  or `optiq_dime/policy.py`, and no gradient proposal implementation was added.
- `.env`, runtime logs, output arrays/checkpoints and local environments are ignored.
- Fixed warmup/UTD/network/proposal settings and environment action rescaling
  use the existing DIME/SB3/JAX path. Only component-aware local diagnostics are
  adapted from `heejoon`; the sampler output and actor objective are preserved.
