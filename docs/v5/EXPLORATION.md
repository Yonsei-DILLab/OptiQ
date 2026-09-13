# v5 Ant exploration follow-up

2026-09-13 user-requested sequence. Each condition runs Ant-v4 seeds 0,1,2,3
to 1M steps; all four seeds finish before the next condition starts.

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

T is the teacher temperature only. The OT regularizer remains .25. There is
no policy-entropy backup term or entropy-dependent acceptance rule.

The 10% uniform option makes one Bernoulli decision per collected action
after the original 5K warmup. On replacement, the entire normalized action
is drawn uniformly from [-1,1]^D and the executed action enters replay.
Replacement continues after annealing ends. Both evaluation modes and TD
next actions bypass this hook. Ordinary Gaussian policy collection remains
the other 90% of actions.

All profiles preserve v5 mean-action student OT, Gaussian teacher/density,
full-row conditional NLL for mu and sigma, plain TD, beta=1, M16/K64,
256x2 networks, initial sigma=.5, Adam and clipping, and paired epsilon=0
evaluation with zero-z and sampled-z. Uniform collection and temperature
annealing reuse existing implementations; no learner equation changes here.

W&B remains **OptiQ/v4-test**, using distinct v5 group/run names and fresh
IDs/directories. Each run records the resolved profile and the actual
temperature, schedule progress, replacement probability and replacement
counts where applicable. Planned jobs are recorded in the instance campaign
manifest; W&B run URLs are created when the workers start.

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_behavior010 \
bash scripts/run_v5.sh 0 --check benchmark=ant

OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_annealing \
bash scripts/run_v5.sh 0 --check benchmark=ant

OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v5_annealing_behavior010 \
bash scripts/run_v5.sh 0 --check benchmark=ant
```

Remove `--check` only for a requested managed training worker. The launcher
checks that uniform probability and presence of annealing match the selected
profile. Original v2/v3/v4 profiles and frozen campaigns retain their sources.

Validation is recorded in [EXPLORATION_VALIDATION.md](EXPLORATION_VALIDATION.md).
