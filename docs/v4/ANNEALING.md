# v4 teacher-temperature annealing

Optional profile `mujoco_v4_annealing` (alias `v4/annealing`) starts teacher
T at 10 and lowers it to .25 in log space, matching the historical fixed
temperature-annealing experiment. For environment step `t`:

```python
progress = clip((t - learning_starts) / anneal_steps, 0, 1)
T = exp((1 - progress) * log(10) + progress * log(.25))
# Return exactly 10 at progress=0, exactly .25 at progress=1.
```

The original 5,000-step random-action warmup remains. The first update uses
t=5001. A 20,000-step anneal reaches .25 at total environment step 25,000;
a 40,000-step anneal reaches .25 at 45,000. T stays .25 thereafter. The
schedule follows environment interactions, regardless of gradient updates
per interaction. It is fixed, stateless, and has no ESS/Q/return feedback.

Only the temperature passed to teacher construction changes. It remains a
dynamic JAX argument, so changing its value does not specialize the compiled
training function at each step. The critic retains plain TD, zero policy
entropy bonus, and no guard. Beta=1, 16x64 OT, epsilon=.25/100 iterations,
conditional Gaussian NLL, 256x2 networks, initial sigma=.5, and both paired
zero-epsilon evaluation modes retain v4 defaults. The schedule is restricted
to plain TD; an absent/disabled schedule preserves constant-temperature runs.

The requested grid has four conditions, each seeds 0,1,2,3 and 1M steps:

| Uniform collection probability | Post-warmup anneal steps | Final T reached at |
|---|---:|---:|
| 0 | 40,000 | 45,000 |
| 0 | 20,000 | 25,000 |
| .1 | 40,000 | 45,000 |
| .1 | 20,000 | 25,000 |

The .1 conditions use the existing [uniform collection hook](BEHAVIOR010.md).
Uniform replacement starts after warmup and continues for the entire run,
including after annealing ends. It bypasses both evaluation modes and TD
target sampling. The p=0 conditions still use ordinary stochastic actor
collection after the random-action warmup.

W&B remains **OptiQ/v4-test**. Record actual `train/temperature`,
`train/temperature_anneal_progress`, `train/temperature_schedule_enabled`,
`train/temperature_final_target`, and `train/temperature_env_steps`.
The last field identifies the environment step whose update consumed T.
Saved configuration preserves initial T=10 and the schedule parameters.
Evaluation metrics remain `eval/zero_z/mean_reward` and
`eval/stochastic_z/mean_reward`, ten episodes each at step 1/every 5K.

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v4_annealing \
bash scripts/run_v4.sh 0 --check benchmark=ant \
  alg.actor.temperature_schedule.anneal_steps=20000 \
  alg.behavior_uniform_probability=0.1
```

Remove `--check` for a managed worker; supply the usual credentials through
`OPTIQ_ENV_FILE`. Register fresh directories and IDs under supervisor. The
requested campaign follows completion of the existing eight constant-T
uniform-collection runs, without changing their frozen sources or workers.
