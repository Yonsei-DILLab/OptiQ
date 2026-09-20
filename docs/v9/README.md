# OptiQ v9: importance-resampled conditional NLL

This branch publishes the implementation running in the
`optiq-resampled-cost50-20260920` campaign. It is based on commit
`71c5ba86cbf63042b227f37cce114143700378c9`; it is not a merge of unrelated
later experimental branches. The existing dirty local checkout and active GPU
experiments are not modified by this release.

## Algorithm

For each replay state:

1. Draw 16 independent standard-normal latent vectors and evaluate the shared
   256x2 actor to obtain conditional Gaussian means and standard deviations.
2. Build a uniform tanh-Gaussian mixture from those components. The teacher-only
   standard-deviation floor is .05. Draw 64 candidates by independently selecting
   a component uniformly and sampling its Gaussian (`exact` proposal mode).
3. Evaluate twin mean-Q and the same proposal's exact finite-mixture density.
   Normalize `exp(Q / temperature - log q)` (beta=1).
4. Perform 16 independent categorical draws WITH replacement using those
   importance weights. This is multinomial resampling, not top-k or stratified
   resampling. Duplicate candidate indices are retained as separate slots.
5. Solve entropic OT between 16 `tanh(mu)` locations and 16 resampled slots,
   with both marginals uniform. Cost is squared Euclidean action distance.
6. Aggregate duplicate slot columns back to the original 64 candidate indices,
   row-normalize, and minimize full-row conditional Gaussian NLL in pre-tanh
   coordinates. This aggregation preserves the slot-wise loss and gradients.
   Both mu and sigma are learned; references and assignments are stop-gradient.

The `transport_target_mode=argmax` inherited config is NOT a hard training
assignment here: conditional NLL uses the full rows. Finite-iteration entropic
OT is generally not a permutation. Uniform slot marginals represent resampled
counts, not the exact original importance weights; the discrepancy is logged.
Finite candidate/resampling error and Gaussian projection error remain.

Critic training is ordinary TD with target twin minimum and no entropy bonus.
Collection and TD next actions use `tanh(mu + sigma * epsilon)`; there is no
separate sigma Q/entropy loss, extra uniform replacement, or added action noise
in this profile. The actor's two heads share the trunk.

## Run

```bash
python run_optiq_dime.py --config-name=mujoco_v9 benchmark=halfcheetah \
  seed=0 alg.actor.temperature=0.25
```

`v9/final` is an equivalent config alias. Release defaults are raw cost,
Sinkhorn epsilon .03 / 50 iterations, teacher temperature .25, no gradient
clipping, 5K warmup, batch 256, UTD 1, learning rates 3e-4, and 1M steps.
The default benchmark is inherited Humanoid; specify `benchmark` explicitly.

Examples matching current explicit temperature choices:

```bash
python run_optiq_dime.py benchmark=hopper alg.actor.temperature=0.05 seed=0
python run_optiq_dime.py benchmark=walker2d alg.actor.temperature=0.1 seed=0
python run_optiq_dime.py benchmark=humanoid alg.actor.temperature=0.1 seed=0
python run_optiq_dime.py benchmark=ant alg.actor.temperature=0.25 seed=0
```

HalfCheetah .05/.25/.5 are comparison settings, not an established optimum.
Other archived sweeps used OT epsilon .01/.05/.1 and/or normalized cost;
override those explicitly. The historical `mujoco_resampled_nll` profile retains
its original normalized-cost, epsilon .05, 30-iteration configuration.
Legacy v5-named configs in this branch do not restore the old 16x64 algorithm;
use the actual historical branch for that comparison.

W&B defaults to `OptiQ/v5-resampled-nll`, group `v9-resampled-nll`.
Authenticate externally using `wandb login` / `WANDB_API_KEY`. No credentials,
checkpoints, datasets, scheduler files or existing run IDs are added by this release.
See the repository's MuJoCo dependency files for installation; CUDA/JAX wheels
must match the target machine. Use a suitable process supervisor for long jobs.

## Evaluation and diagnostics

Every 5K steps evaluate 10 episodes per mode:

- `eval/stochastic_z/mean_reward`: fresh Gaussian z each action, Gaussian
  epsilon=0, action=`tanh(mu(s,z))`.
- `eval/zero_z/mean_reward`: z=0 and epsilon=0.

These are NOT full Gaussian collection-policy returns. Checkpoints are saved
every 50K steps. Actions are normalized for replay and unscaled for env.step.

Useful metrics include `train/actor_std_mean`, sigma lower/upper-bound fractions,
`mean_action_spread`, `resample_unique_candidates`, `resample_mass_tv`,
`ot_slot_row_ess`, `ot_unique_row_ess`, `ot_unique_row_max`,
`ot_slot_row_marginal_error`, `ot_slot_col_marginal_error`, and `ot_row_overlap`.
Duplicate slots can inflate slot entropy without representing distinct actions;
use unique-candidate diagnostics alongside marginal residuals.

## Validation and provenance

The 166 Python/YAML files in the frozen runtime source manifest were checked
against the source snapshot before release; all matched. The algorithm.py,
policy.py, resampled_ot.py and transport.py implementations are copied without
algorithm changes. Release-only additions are v9 configs, documentation/tests,
and selection of the v9 default entry in the launcher.

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_v9_resampled.py -q
```

Tests check categorical frequencies, duplicate aggregation of NLL and gradients,
teacher/plan stop-gradient behavior, uniform column masses, configuration
aliases for all five environments, and a production actor update changing both
mu and sigma heads. Release validation on 2026-09-20: **3 passed** on CPU using
the existing campaign runtime (JAX 0.4.33, Flax 0.9.0); only dependency
deprecation warnings were emitted. These are numerical/configuration/update
checks, not new benchmark runs.
