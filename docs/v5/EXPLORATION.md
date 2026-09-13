# v5 Ant exploration follow-up

Historical 2026-09-13 user-requested sequence. All twelve runs completed 1M
steps under the original epsilon=.25 / gradient clip=2 defaults. This queue is
finished. The active no-clip grid is documented in [EXPERIMENTS_KO.md](EXPERIMENTS_KO.md).

| Order | Profile | Teacher temperature | Extra uniform collection |
|---|---|---|---:|
| 1 | `mujoco_v5_behavior010` | .25 fixed | 10% |
| 2 | `mujoco_v5_annealing` | 10 → .25 over 40K after warmup | 0% |
| 3 | `mujoco_v5_annealing_behavior010` | Same 40K schedule | 10% |

Aliases are `v5/behavior010`, `v5/annealing`, and `v5/annealing_behavior010`.
The fixed-temperature `mujoco_v5` default remains unchanged.

Annealing follows the existing v4 schedule:

```python
progress = clip((env_steps - 5000) / 40000, 0, 1)
T = exp((1-progress)*log(10) + progress*log(.25))
# Exact endpoints: T=10 before/at 5K, T=.25 from 45K onward.
```

T is the teacher temperature only. The completed runs used OT epsilon=.25;
these optional profiles now inherit the updated v5 default epsilon=.1 and no
gradient clipping. There is
no policy-entropy backup term or entropy-dependent acceptance rule.

The 10% uniform option makes one Bernoulli decision per collected action
after the original 5K warmup. On replacement, the entire normalized action
is drawn uniformly from [-1,1]^D and the executed action enters replay.
Replacement continues after annealing ends. Both evaluation modes and TD
next actions bypass this hook. Ordinary Gaussian policy collection remains
the other 90% of actions.

All profiles preserve v5 mean-action student OT, Gaussian teacher/density,
full-row conditional NLL for mu and sigma, plain TD, beta=1, M16/K64,
256x2 networks, initial sigma=.5, Adam, and paired epsilon=0
evaluation with zero-z and sampled-z. Uniform collection and temperature
annealing reuse existing implementations; no learner equation changes here.

The completed exploration runs use **OptiQ/v5-test**, with distinct v5 group/run names and fresh
IDs/directories. Each run records the resolved profile and the actual
temperature, schedule progress, replacement probability and replacement
counts where applicable. Planned jobs are recorded in the instance campaign
manifest; W&B run URLs are created when the workers start.

The examples below resolve the **historical** exploration recipe with explicit
epsilon=.25 and clip=2 overrides. Omit those two overrides to inspect the new
default recipe. Changing defaults does not modify the completed runs or their results.

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_behavior010 \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-test alg.actor.sinkhorn_epsilon=.25 alg.optimizer.ac_grad_norm=2.0

OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_annealing \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-test alg.actor.sinkhorn_epsilon=.25 alg.optimizer.ac_grad_norm=2.0

OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_annealing_behavior010 \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-test alg.actor.sinkhorn_epsilon=.25 alg.optimizer.ac_grad_norm=2.0
```

Remove `--check` only for a requested managed training worker. The launcher
checks that uniform probability and presence of annealing match the selected
profile. Original v2/v3/v4 profiles and frozen campaigns retain their sources.

Validation is recorded in [EXPLORATION_VALIDATION.md](EXPLORATION_VALIDATION.md).

On 2026-09-13 the user changed the project to `v5-test`. All twelve jobs were
re-registered with this explicit override; the four active, incorrectly routed
runs were restarted with fresh IDs. Completed meanOT baseline records are
re-uploaded without retraining. Original records remain as provenance.
Learning source and hyperparameters are unchanged.
