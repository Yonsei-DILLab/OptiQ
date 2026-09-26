# GMM40 fixed-energy experiment

This profile is separate from MuJoCo and maze experiments. There is no learned
critic, replay buffer, environment collection, UTD, or DACER exploration.
The actor receives a constant dummy state and fresh 2D standard-normal latents.
Training queries only Q(x) = log p_GMM40(x), never ground-truth samples.

| Setting | Value |
|---|---|
| Network | 256 x 3, GELU; 133,636 parameters |
| Paper comparison | N=M=256 |
| Sample-count ablation | N=M in {64,128,256,512} |
| Update budget / seeds | 100,000 actor updates / 0,1,2,3 |
| Batch / Adam learning rate | 256 independent clouds / 3e-4 |
| Temperature / density correction | 1 / 1 |
| Mean-head initializer | variance scale 16, fan-average uniform |
| Actor log-sigma bounds / initial | [-5,-3.5] / -4 |
| Teacher-only sigma floor | 0.05 |
| Evaluation | 10,000 full-policy and 10,000 mean-only samples |

Actions live in [-1,1]^2 and physical coordinates are x=40a. Centers are
tanh-bounded; conditionals are box-truncated Gaussians. The proposal floor is
used consistently in candidate sampling and proposal density, not in the actor
likelihood or full-policy sampling. Candidates and importance weights are
stop-gradient targets. The objective is direct marginal NLL, without transport.
The physical-coordinate Jacobian contributes a constant that cancels in weights.

The numerical target definition is bundled in `target.json`: the DiKL seed-0
40-component benchmark, restricted to the physical box for reference sampling.
No baseline implementation or historical experiment package is needed at runtime.
The SHA256 of Python `json.dumps(values, sort_keys=True).encode()`, where `values`
contains only `means`, `std`, and `weights`, is:
`a22a67f6f37d782fd75961e75f5df13c424fda9624245c99efe93862c929e87c`.
The bounded target mass is approximately 0.98267287. Q uses the original mixture
log density; bounded-reference normalization is state-independent.

Evaluate and checkpoint at updates 0,100,500,1000,2500,5000, then every 10,000
through 100,000. Reference seed: 20260917. Policy evaluation seed: 900000+seed.
Evaluation does not consume the training RNG. Full-policy output is primary;
mean-only output is supplementary. Keep both views separate.

MMD is sqrt(max(unbiased MMD2,0)), using the first 2,048 samples and five RBF
bandwidths {1,2,5,10,20} in physical coordinates. Coverage counts nearest target
components within 3 sigma, with occupancy at least max(10, 10% of reference
occupancy adjusted for sample count). This is component coverage, not an exact
count of density local maxima. Other retained metrics include sliced W2 and
high-density fraction. Aggregate four training seeds with sample standard
deviation; do not pool N/M settings or full-policy and mean-only results.

Equal optimizer budgets are not equal query or compute budgets:
training Q queries = updates * batch * M. The original sample-count campaign
used RTX 4090 hardware. This release does not claim timing equivalence across
devices or a fresh 100K reproduction from the short validation tests.

## Commands

From the repository root, after installing the shared requirements:

```bash
python -m gmm40.run --seed 0 --nm 256 --output outputs/gmm40-N256-s0
# Repeat seeds 0..3; ablation changes --nm to 64, 128, 256, or 512.
# Optional tracking, without a hardcoded account:
python -m gmm40.run --seed 1 --nm 256 --output outputs/gmm40-N256-s1 --wandb
```

`--wandb` defaults to offline mode. Pass `--wandb-mode online` after logging in
for online tracking. `--steps`, `--batch`, and `--eval-samples` support short
validation only; changing them does not reproduce the paper protocol.
Existing output directories are never overwritten. `--resume` restores the
GMM40 actor optimizer, RNG, and update count; use a new output directory and the
same settings. MuJoCo checkpoint semantics are separate.
