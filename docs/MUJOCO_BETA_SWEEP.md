# Ant-v4 / Humanoid-v4: OptiQ actor with DIME critic

The `heechan` branch starts at `critic-dime` commit
`8b8fee13b2cbc4d90183d7b803fce033eadf0a19`. It retains DIME's JAX
distributional CrossQ critic and the one-step implicit OptiQ actor. No
Q-gradient, covariance, Gamma, or other proposal-family variants are included.
Only the component-aware sampling diagnostics were adapted from `heejoon`;
the existing isotropic samplers and actor objective are retained.

## Protocol

| Setting | Value |
| --- | --- |
| Environments | Ant-v4, Humanoid-v4 |
| Steps | Ant: 3,000,000; Humanoid: 5,000,000 |
| Seeds | 1, 2, 3, 4, 5 (override with `--seeds`) |
| Sampling | `stratified`, `exact` |
| Fixed density beta | 0.1, 0.25, 0.5, 0.75, 1.0 |
| Adaptive beta | Off |
| Actor | 256 x 256 x 256, GELU, one-step implicit policy |
| Critic | Two 2048 x 2048 ReLU critics, 101 categorical atoms |
| Critic support | [-1600, 1600], Gym setting approved for this experiment |
| Critic normalization | Batch renormalization, momentum 0.99, warmup 100,000 |
| Critic distribution entropy coefficient | 0.005 |
| Actor/critic LR | 0.0003; Adam betas (0.9, 0.999)/(0.5, 0.999) |
| Replay / batch / warmup | 1,000,000 / 256 / 5,000 |
| UTD / policy delay / discount | 2 / 1 / 0.99 |
| Critic tau / policy tau | 1.0 / 1.0 |
| Max-entropy alpha | 0.0, as in the existing OptiQ+DIME hybrid |
| Policy samples / candidates | 16 / 80 |
| Proposal std / local clip | 0.1 / 0.15 in normalized action coordinates |
| Include anchor | True, matching the supplied run configuration |
| Q temperature / source Q | 0.25 / mean of twin expected Q values |
| Sinkhorn epsilon / iterations | 0.05 / 30 |
| OT target | Argmax of each conditional transport row, MSE distillation |
| TD noise std / clip | 0.2 / 0.5 |
| Evaluation | Stochastic, 10 episodes at step 1, every 5,000 steps, and final step |
| Checkpoints | Initial trained state, every 50,000 steps, and final actor/critic |

The supplied Dog configuration is used except for the approved Gym critic
support, environment budgets, and the fixed-beta/sampling sweep axes. The
network, optimizer and actor proposal hyperparameters are not tuned separately
for the two samplers. v3 and v5 are outside this experiment.

## What the sampling comparison means

Actor draws `x_i = mu(s, z_i)` define a uniform mixture of exactly box-truncated
Gaussian components. Proposal density is evaluated using these same centers,
scales and bounds; it is not a KDE fitted to the newly generated candidates.

- `stratified`: draw the same number of random candidates from every component.
- `exact`: choose each random candidate's component independently and uniformly,
  then sample that component. This mode was already implemented in `critic-dime`.

Both modes use `softmax(Q/T - beta * log(q))` and then the same OT/actor update.
Beta here controls inverse-density correction; it is not the Q temperature or
an adaptive parameter. Stratified component allocation with mixture weights
is a deterministic-mixture importance sampling scheme, not inherently an
incorrect density calculation. This experiment tests its learning behavior
against IID mixture allocation.

With the supplied `include_anchor=true`, 16 of the 80 candidates are fixed
copies of actor centers and 64 are random proposals. Only the random proposals
in `exact` are IID mixture draws. The anchor weights retain the original
continuous-density heuristic; do not label the entire candidate cloud as IID
or claim exact global Boltzmann sampling. Local proposal support, finite
samples, beta < 1 and argmax OT distillation also affect the resulting target.
For a separate pure-random-candidate comparison, pass
`alg.actor.include_anchor=false` to every arm: both then use 80 random candidates.
Run/group names include the anchor setting to prevent accidental pooling.

Local Q diagnostics now group by the component that actually generated each
candidate. The previous reshape-based grouping was correct only for stratified
allocation. This change does not change sampling or the actor loss.

## Install and authenticate

```bash
cd /workspace/OptiQ
bash scripts/setup_mujoco_env.sh
```

The default isolated environment is `/workspace/.venv-optiq-mujoco`.
Set `OPTIQ_VENV` to choose another setup location, and `OPTIQ_PYTHON` to select
its interpreter when launching. The complete resolved dependencies are in
`requirements-mujoco.lock`; direct constraints are in `requirements-mujoco.in`.
Python 3.11, JAX/JAXlib 0.4.33, Flax 0.9.0, Gymnasium 0.29.1 and MuJoCo 2.3.7
are pinned. PyTorch 2.4.1 uses CPU for SB3 buffers; actor, critic and OT execute
on JAX CUDA. A PyTorch `device=cpu` field does not identify the JAX backend.
Training explicitly rejects a JAX CPU fallback.

Store W&B credentials in the repository `.env` (ignored by Git), or specify
`OPTIQ_ENV_FILE=/absolute/path/to/.env`. Existing shell variables take precedence,
followed by the explicit file, repository `.env`, then workspace `.env`.
`WANDB_PROJECT` and `WANDB_ENTITY` are honored. The default project is
`optiq_dime_mujoco_v4`; an unset entity uses the authenticated account.
The launch path requires online W&B and fails on initialization/authentication
errors. It never prints dotenv contents or saves credentials in configuration.

W&B config records `env_name`, environment observation/action shapes and action
bounds, sampling, fixed beta, anchor/random counts, resolved hyperparameters,
Git SHA/dirty state, package versions, command and actual JAX GPU devices.
Metrics use `env_steps` as the x-axis. Evaluation uses separate action/environment
random streams and restores the training policy's RNG state afterward.

## Inspect or run

```bash
# Print all 100 commands without starting training.
bash scripts/run_mujoco_beta_sweep.sh --list

# First task: Ant-v4, stratified, beta=0.1, seed=1.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mujoco_beta_sweep.sh --task 0

# Inspect a smaller seed set (IDs are regenerated for the selected seeds).
bash scripts/run_mujoco_beta_sweep.sh --list --seeds 1,2

# Foreground worker 0 of 4 on an allocated GPU. Workers partition the task table.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mujoco_beta_sweep.sh --worker 0 4

# Optional anchor-free variant, applied consistently to all selected arms.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mujoco_beta_sweep.sh --worker 0 4 \
  alg.actor.include_anchor=false
```

Default task IDs: 0-24 Ant/stratified; 25-49 Ant/exact;
50-74 Humanoid/stratified; 75-99 Humanoid/exact. Within each block beta changes
every 5 tasks and seed changes fastest. Matrix axes cannot be silently changed
by trailing Hydra overrides. Other overrides, such as a short validation budget,
are written to the resolved W&B config.

For long-running work on this Vast instance, review and install
`deploy/supervisor/optiq-mujoco-beta.conf` in `/etc/supervisor/conf.d/`, then run
`supervisorctl reread` and `supervisorctl update`. The template has
`autostart=false`; explicit start is `supervisorctl start 'optiq-mujoco-beta:*'`.
It assigns one sequential worker to each of four GPUs. Adjust paths/GPU allocation
before use on another instance. GPU locks prevent concurrent sweep workers on
the same visible GPU identifier. Other unrelated GPU processes are not managed.

Workers stop on the first failed task. Starting a worker again starts its assigned
tasks anew; there is no implicit resume or completion skipping. Each invocation
has a unique output directory and W&B run, preserving earlier results. Inspect
completed results and use `--task` for precise reruns. Checkpoints contain
actor/critic train states for analysis; they are not full replay/RNG resume files.

## Validation and outputs

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mujoco_beta_sweep.sh --task 0 --seeds 1 \
  total_steps=12 alg.learning_starts=8 num_eval_episodes=1 eval_at_start=false \
  log_interval=4 checkpoint_interval=10 wandb.job_type=smoke-test \
  output_root=outputs/validation

CUDA_VISIBLE_DEVICES=0 /workspace/.venv-optiq-mujoco/bin/python -m pytest -q tests
```

Validation tests also create an online W&B run. They cover mixture sampling vs
density consistency, component allocation/anchors, all ten fixed-beta/sampling
actor updates, all 100 task configurations, v4 environment shapes/bounds,
time-limit bootstrap masks, and evaluation RNG isolation/result persistence.
Short training validation must not be used as a learning-performance comparison.

Each run directory stores `config.json`, CSV/TensorBoard metrics, per-episode
`evaluations.npz`, actor/critic checkpoints, and `completed.json` on successful
training completion. W&B artifacts include the config, evaluations and final
actor/critic checkpoint; intermediate checkpoints remain local. No external
offline dataset or pretrained model is required. MuJoCo supplies online data.

For analysis, filter W&B to `job_type=train`, compare sampling and beta within
each environment and anchor setting, and report individual seeds plus aggregate
return curves and uncertainty. ESS and Q diagnostics explain behavior; they do
not replace evaluation return. The base `main`/`dev` OptiQ critic is not used.
