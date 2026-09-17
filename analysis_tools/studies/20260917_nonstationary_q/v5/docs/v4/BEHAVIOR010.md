# v4: 10% uniform collection

Profile: `mujoco_v4_behavior010`, alias `v4/behavior010`.
This reuses the existing `OptiQDIME._sample_action` exploration hook from the
earlier 10% behavior experiments. The default `mujoco_v4` profile remains at
`alg.behavior_uniform_probability=0.0`.

Keep the original 5,000-step random-action warmup. Subsequently, at each
collection step, independently replace the entire action vector with
probability .1 by a uniform sample from the environment's valid action box:

```python
a = sample_current_actor(s)  # includes latent z and conditional Gaussian epsilon
if behavior_rng.uniform() < 0.1:
    a = uniform(-1, 1, shape=action_dim)  # normalized action coordinates
env.step(unscale(a))
replay.add(s, a, reward, s_next, done)    # store the actual executed action
```

The decision applies to the whole vector, not separately to its coordinates.
It is a random 10% probability, not an exact quota. Uniform actions replace
actor actions; there is no additive noise or interpolation with the actor.
The behavior RNG is separate from the actor, replay and warmup RNG streams.

Collection changes the replay data. TD target actions and the actor's
teacher/OT/NLL use their original sampling paths. Both evaluations also
bypass this collection hook: `zero_z` and `stochastic_z`, each with epsilon=0.
There is no new entropy bonus, guard, Gaussian exploration controller or
change to the learnable conditional sigma. Initial sigma remains .5.
Actor and critic remain 256x2. The existing 1M budget and 5K/10-episode
evaluation schedule apply unless explicitly overridden.

## Launch configuration

This profile routes to **`OptiQ/v4-test`** and marks run names and groups with
`behavior010`. It defaults to T=.1 and supports the requested T=.25:

```bash
# Inspect without training or W&B writes (seed can be 0, 1, 2, or 3).
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_CONFIG=mujoco_v4_behavior010 \
bash scripts/run_v4.sh 0 --check benchmark=ant alg.actor.temperature=0.1

# Worker command: use under supervisor for a long-running experiment.
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
OPTIQ_ENV_FILE=/root/OptiQ/.env \
OPTIQ_CONFIG=mujoco_v4_behavior010 \
bash scripts/run_v4.sh 0 benchmark=ant alg.actor.temperature=0.25
```

The equivalent direct entry is `python run_optiq_dime.py
--config-name=mujoco_v4_behavior010 benchmark=ant seed=0`.
Change the seed to select another independent run. Use fresh output paths
and a campaign-specific group when scheduling repeated comparisons.

W&B records `rollout/behavior_uniform_probability`,
`rollout/behavior_uniform_count`, `rollout/behavior_action_count` and
`rollout/behavior_uniform_fraction`. Counts exclude the initial warmup;
the fraction is the observed cumulative replacement rate. Evaluation metrics
retain `eval/zero_z/mean_reward` and `eval/stochastic_z/mean_reward`.

This addition provides an opt-in configuration. Existing frozen training
sources, running baseline experiments and their temperature queue are intact.
