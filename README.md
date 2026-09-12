# OptiQ v4

The v4 default uses **256x2 actor and critic networks** and evaluates both
`tanh(mu(s,0))` and `tanh(mu(s,z))`, with fresh `z ~ N(0,I)` per action in the
second mode. Both set epsilon to zero. Collection and TD use the full Gaussian
policy. See [v4 specification and commands](docs/v4/PSEUDOCODE.md).

Run `bash scripts/run_v4.sh 0 --check benchmark=hopper` to inspect the config.
`python run_optiq_dime.py` now defaults to `mujoco_v4`.

## Historical v3 reference

v3 uses a **plain twin-min TD backup**, a continuous-latent conditional Gaussian
policy, beta=1 proposal density correction, and full 16×64 OT Gaussian NLL.
There is no policy entropy bonus, IDAC entropy evaluation, soft-score acceptance
check, or optimizer rollback in the default v3 update path.

The teacher Boltzmann temperature (T=0.1), entropic OT regularizer (epsilon=0.25),
and Gaussian likelihood normalization remain part of the actor construction.
Teacher density is used for importance correction. These are distinct from
adding policy entropy to the TD target or an actor objective.

- **[v3 implementation pseudocode (Korean)](docs/v3/PSEUDOCODE.md)**
- [Canonical configuration](configs/v3/final.yaml)
- [Validation results and scope](docs/v3/VALIDATION.md)
- [v2 baseline pseudocode](docs/v2/PSEUDOCODE.md)

## Run or inspect

The pinned environment is shared with v2; installation instructions are in
[the reproduction guide](docs/v2/REPRODUCIBILITY.md). On this instance use
`/root/.venv-optiq-mujoco/bin/python`. The v3 worktree is `/root/OptiQ-v3`;
`/root/OptiQ` remains the baseline v2 checkout.

```bash
scripts/run_v3.sh --list
scripts/run_v3.sh 0 --check
scripts/run_v3.sh 0 --check benchmark=ant
```

Checks compose and validate the requested configuration without starting
training or W&B. Pass `--config-name=mujoco_v3` to reproduce v3 on this
branch. `mujoco_v3` and `v3/final` resolve to the same self-contained config.
The launcher enforces plain TD, zero policy entropy, no guard, conditional
mixture, and full OT NLL. Use `OPTIQ_PYTHON` and `OPTIQ_ENV_FILE` to specify the
Python interpreter and credential file when needed.

For a requested GPU experiment, run the wrapper under supervisor using the
[service template](deploy/supervisor/optiq-v3.conf) and
`scripts/supervisor_v3.sh`. The template has autostart disabled. No v3 service
is installed or started by creating this branch or running validation.

Baseline numerical settings remain: 256×3 networks, batch 256, warmup 5K,
16 student latents, 64 teacher candidates, Sinkhorn epsilon=.25/100 iterations,
conditional teacher std floor=.05, Adam LR=3e-4, global gradient clipping=2,
no LayerNorm, and no extra uniform exploration. Default runs use 1M environment
steps per seed without performance-based early stopping. Teacher uses live
mean-Q; TD uses target min-Q. Every scheduled actor update is applied.

Comparisons use `OptiQ/optiq_mujoco_v2_confirmation` with separate v3 group/run
names and fresh IDs. Outputs go to `../optiq-experiments/v3_td/outputs/` under
a fresh directory per run. Keep credentials and outputs outside tracked source.

## Validation

See [the validation record](docs/v3/VALIDATION.md) for exact checks. Short
Humanoid/Ant loops exercise the real networks, TD, OT NLL, timeout handling and
checkpoint restoration. They do not establish v3 performance. No full v3
benchmark result is claimed.

Existing explicit v2 configs retain their numerical behavior and regression
coverage. The frozen v2 launcher also verifies original source hashes, so use
that launcher from the original `/root/OptiQ` checkout, not this modified branch.
The v2 docs and results under `docs/v2` describe the baseline, not v3.
