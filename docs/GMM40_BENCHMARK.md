# GMM40: current OptiQ as an unconditional generator

## Latest requested rerun: native coordinates, no factor 50

Use `scripts/run_gmm40_native.sh`. The user requested **direct original GMM
coordinates**, so `coordinate_scale=1`, `proposal_std=8`, and `x=g(z)`.
There is no output scaling, normalization, clipping or proposal truncation.
The existing unbounded OptiQ actor path handles these units directly; no
additional change to the common algorithm is needed.

N=256, one anchor + four random candidates per center, one 256×1280 OT,
T=0.25 fixed, beta=0.1, 256×3 GELU, Adam lr=0.0003, and 30k updates per
seed 0/1/2 remain. MSE is now computed in original GMM units. Actor
initialization, learning rate and loss are not adjusted to compensate for
the unit change. Results go to `outputs/gmm40_native/runs`; previous runs
remain available. The queue template uses free GPUs 0/1/2 for seeds 0/1/2.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40_native.sh --seed 0
scripts/evaluate_gmm40.sh outputs/gmm40_native/runs \
  outputs/reports/gmm40_native_final \
  --group gmm40-optiq-native-256x1280-anchor-std8-T0.25-beta0.1-30k
```

## Previous rerun: unbounded 256×1280, anchor=true, physical sigma 8

Use `scripts/run_gmm40_256_anchor_std8_unbounded.sh`. The user requested removal
of truncation for GMM40. This removes **both proposal perturbation/action bounds
and generator output clipping** in the GMM run: candidates come from ordinary
Gaussian kernels on R², and evaluation uses `x=50*actor(z)` without clipping.
The scale 50 remains a coordinate conversion, not an output bound.

N=256, R=5 (one anchor + four random candidates per center), one 256×1280 OT,
physical sigma=8 (normalized 0.16), T=0.25 fixed, beta=0.1 and 30k updates for
seeds 0/1/2 remain. The new output root is
`outputs/gmm40_256_anchor_std8_unbounded/runs`.

The common `OptiQDIME.update_actor` now has an optional static argument
`unbounded_actions=False`. Only the GMM runner enables it. This selects a new
`GaussianKDE` whose sampling and log density use the same unbounded mixture,
and bypasses clipping of actor centers. OT, source weights, stop-gradients,
argmax distillation and MSE are unchanged. RL calls retain the default bounded
path. Tests compare that bounded update with the actual committed implementation,
verify unbounded density against PyTorch, and exercise samples outside the old
limits and evaluation outputs beyond +/-50.

`proposal_clip=0.5` remains an ignored compatibility argument for this path;
the saved effective proposal clip and output bound are explicitly null.
Removing the bounds also changes random sampling from inverse-CDF truncated
draws to ordinary Gaussian draws. Seeds match, but realized noise samples do not.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40_256_anchor_std8_unbounded.sh --seed 0
scripts/evaluate_gmm40.sh outputs/gmm40_256_anchor_std8_unbounded/runs \
  outputs/reports/gmm40_256_anchor_std8_unbounded_final \
  --group gmm40-optiq-256x1280-anchor-std8-unbounded-T0.25-beta0.1-30k
```

## Previous rerun: 256×1280, anchor=true, physical sigma 8

Use `scripts/run_gmm40_256_anchor_std8.sh`. The user specified **256 KDE
centers, each contributing its own anchor plus four random candidates**:
`N=256`, `R=5`, `batch_size=1`, one **256×1280 OT** per update. The 1,280
candidates comprise 256 deterministic anchors and 1,024 random KDE draws.
Physical KDE sigma is **8** (variance 64), normalized sigma **0.16**.
The existing scale `x=50u`, output bounds and normalized perturbation clip
0.5 (physical 25) are retained. GMM component sigma is unchanged.

T=0.25 fixed, beta=0.1, Sinkhorn epsilon 0.05 and 30 iterations, row argmax,
MSE, actor/optimizer and seeds 0/1/2 each with 30k updates are unchanged.
The adapter now forwards `--include-anchor` to the existing actor update;
`optiq_dime/algorithm.py`, `policy.py` and `transport.py` are not modified.
The no-anchor flag remains the benchmark default. Original-code validation
requires `R>=2` when anchors are enabled, since at least one random draw per
center must remain.

This run makes **38,400,000 training target-density evaluations/seed**,
including 7,680,000 anchor evaluations and 30,720,000 random-candidate
evaluations. Final metrics use 10,000 direct g(z) outputs, without adding
KDE noise or applying OT to the evaluation samples.

Both anchors and random candidates use the unchanged legacy score
`log p(x)/T - beta*log q_KDE(u)`. Because anchors are deterministic centers,
the entire candidate set is not sampled randomly from the continuous KDE.
The no-anchor idealized `p^4 q_KDE^0.9` interpretation below does not apply
to the entire anchor-containing set. No new importance-correction rule is
introduced. Sigma, anchor inclusion and candidate count changed together;
this is not an isolated anchor ablation against previous runs.

`scripts/supervisor_gmm40_256_anchor_std8.conf.example` runs seeds 0→2 on
GPU 0 and seed 1 on GPU 1. Previous results are preserved; these outputs go
under `outputs/gmm40_256_anchor_std8/runs`.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40_256_anchor_std8.sh --seed 0
scripts/evaluate_gmm40.sh outputs/gmm40_256_anchor_std8/runs \
  outputs/reports/gmm40_256_anchor_std8_final \
  --group gmm40-optiq-256x1280-anchor-std8-T0.25-beta0.1-30k
```

## Previous rerun: 2048×2048, physical sigma 4

Use `scripts/run_gmm40_2048.sh`: **2,048 generator/KDE centers, 2,048 random
candidates, one 2048×2048 OT problem per update** (`batch_size=1`, `N=2048`,
`R=1`). Physical sigma stays **4** (variance 16, normalized sigma **0.08**),
with normalized clip 0.5. T=0.25 fixed, beta=0.1, no anchors, the actor,
optimizer, and 30k updates/seed remain unchanged. Final evaluation still uses
10,000 direct generator samples.

This makes **61,440,000 training target-density evaluations/seed**, eight
times the 256×256 run. The OT matrix has 64 times as many entries. Comparisons
therefore use equal update counts, not equal sample or compute budgets.

`scripts/supervisor_gmm40_2048.conf.example` queues seeds 0→2 on GPU 0 and
seed 1 on GPU 1. Results go to `outputs/gmm40_2048_std4/runs`; the earlier
256×256 and 16×64 results remain in their separate directories.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40_2048.sh --seed 0
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  /workspace/.venv-optiq-no-anchor/bin/python -m benchmarks.gmm40.evaluate \
  --runs-root outputs/gmm40_2048_std4/runs \
  --group gmm40-optiq-2048x2048-std4-T0.25-beta0.1-30k \
  --output-dir outputs/reports/gmm40_2048_std4_final
```

To run that evaluation and then build/upload the Korean report and training
figures in one command after all three seeds finish:

```bash
scripts/evaluate_gmm40.sh outputs/gmm40_2048_std4/runs \
  outputs/reports/gmm40_2048_std4_final \
  --group gmm40-optiq-2048x2048-std4-T0.25-beta0.1-30k
```

Use a fresh report directory each time. `--baseline-report` can reuse a prior
common evaluation after the evaluator verifies protocol, source and sample hashes.

## Previous rerun: 256×256, physical sigma 4

Use `scripts/run_gmm40_256.sh`. This earlier instruction superseded the
intermediate 2,000-point discussion: **256 generator/KDE centers, 256 candidates,
one 256×256 OT problem per update** (`batch_size=1`, `N=256`, `R=1`). Every
candidate is a random draw from its KDE component; the component centers are
not inserted into the candidate set. Physical sigma is **4** (variance 16),
implemented as normalized sigma **0.08**. The clip remains 0.5 normalized.
T=0.25 fixed, beta=0.1, optimizer, model and 30k updates/seed are unchanged.
The common original actor update is called directly; no separate KDE adapter
or change to the RL algorithm was needed.

The 256×256 run makes **7,680,000 training target-density evaluations/seed**.
It differs from the first 16×64/batch-256/sigma-10 run in both OT geometry and
update batch size, so their 30k budgets are not matched in sample count.
Final evaluation still uses 10,000 generator samples: the user's sample-count
request concerns KDE/OT, not the metric sample count.

The queue template `scripts/supervisor_gmm40_256.conf.example` runs seeds 0→2
on GPU 0 and seed 1 on GPU 1. Outputs are separate under
`outputs/gmm40_256_std4/runs`. Previous completed results are preserved.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40_256.sh --seed 0
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  /workspace/.venv-optiq-no-anchor/bin/python -m benchmarks.gmm40.evaluate \
  --runs-root outputs/gmm40_256_std4/runs \
  --group gmm40-optiq-256x256-std4-T0.25-beta0.1-30k \
  --output-dir outputs/reports/gmm40_256_std4_final
```

The remainder records the original port and first-run protocol for provenance.

This benchmark ports the GMM target and common evaluator from OptiC into
`heechan-no-anchor`. It trains a new unconditional `g(z)` from scratch by
calling **the existing `OptiQDIME.update_actor`**, without changing the RL
implementation or loading Ant/HalfCheetah policy checkpoints.

## Adapter and the requested setting

- `ImplicitActor` receives an empty observation and a 2-D standard Gaussian
  latent. Its existing 256×256×256 GELU architecture and initialization remain.
- Normalized output `u = clip(actor(z), -1, 1)` maps to `x = 50*u`.
  The generator therefore has a bounded output domain; the reference GMM is
  evaluated in its original, unbounded coordinates.
- Exact `log p(x)` replaces the learned critic. Two identical oracle outputs
  satisfy the existing twin-critic interface; there is no critic training,
  replay buffer, or environment interaction. The existing stop-gradient on
  Q/proposals/transport remains, so this is not an energy-gradient variant.
- Batch size 256 means 256 independently sampled OT problems per update.
  Each has 16 actor outputs and 64 random candidates, four per KDE component.
- KDE sigma 0.2 and perturbation clip 0.5 in normalized coordinates correspond
  to 10 and 25 in physical coordinates. The existing action bounds truncate it.
- No anchors; density beta 0.1; Sinkhorn epsilon 0.05, 30 iterations;
  row argmax targets; pointwise MSE; Adam lr 0.0003, betas 0.9/0.999.
- **T=0.25 fixed**, as explicitly requested. There is no annealing.
- Prespecified budget: seeds 0, 1, 2; 30,000 actor updates per seed. No best
  checkpoint selection. Each run makes 491,520,000 training target-density
  point evaluations, excluding evaluation-only calls.
- Evaluation every 1,000 updates, 10,000 direct generator samples. Evaluation
  latents are fixed per seed and independent of training random keys.

These settings are an evaluation of the existing RL actor rule on a known
density, **not an exact GMM sampler configuration**. The candidate weights are

```
w_i ∝ exp(log p(50*u_i) / 0.25 - 0.1 * log q_KDE(u_i))
```

For a fixed proposal, the corresponding ideal weighted candidate measure is
proportional to `p(x)^4 * q_KDE(x)^0.9` on its support, before finite-sample OT
and argmax distillation. It is not generally `p(x)` or just `p(x)^4`.
The score is still measured against the original GMM40, per the user request.

## Target identity and imported code

`benchmarks/gmm40/target_torch.py` is the verbatim OptiC
`DiKL/energy/mog40.py`. The canonical target has 40 equally weighted Gaussian
components in 2-D, means drawn with PyTorch seed 0 in [-40,40]^2, and component
standard deviation `softplus(1) ≈ 1.31326`. All network seeds use this same target.
The JAX oracle is checked numerically against its PyTorch log density.

`benchmarks/gmm40/metrics.py` comes from OptiC `DiKL/benchmark_eval.py`.
Only its tiny nearest-mode helper is inlined to remove a dependency on the
unrelated OptiC algorithm module. An equivalence test compares the two
evaluators when the original checkout is available. Original source hashes,
commit and license are retained in `source_provenance.json` and
`LICENSE-OptiC-DiKL` next to the port. OptiC's working files are not edited.

## Evaluation and literature comparison

The primary published GMM metric is mean `log p_d(x)` in DiKL Table 1:
True -6.85, FAB -10.74, iDEM -8.33, DiKL -7.21. That table reports the methods
covering all modes; R-KL has no entry. The paper describes a 2.5-hour GMM
training budget and uses a different generator architecture. Neither compute
budget nor architecture is matched by this port.

Sources:

- [DiKL paper, Table 1 and Appendix D.1](https://arxiv.org/html/2410.12456v2)
- [Official code and released sample links](https://github.com/jiajunhe98/DiKL)

The common evaluation also reports absolute error to the same GT mean,
coverage, occupancy TVD, energy W1/KS, sliced W2, low-density fraction, exact
empirical sample W2, and energy TVD. These additional GMM numbers are a
**new re-evaluation of released samples**, not values quoted from the GMM table.

All methods use 10,000 samples and one deterministic CPU target draw with
reference seed 20260821. FAB's larger release is subsampled with an independent,
fixed RNG; its full-release log p is also retained. CPU/GPU PyTorch generators
can produce different reference draws, so historical OptiC reference means
need not equal this run's mean even with the same integer seed.

Exact sample W2 uses ten repeats of 2,000-point sampling without replacement
from the generated/reference pools. Equal uniform sample weights permit an
exact SciPy linear assignment instead of adding a POT dependency. Its cost
uses the source evaluator's squared `torch.cdist` convention. Repetition SD
describes evaluation subsampling, **not independent training seeds**. A
separate target-vs-target sample set exposes the finite-sample metric floor.

Energy histogram TVD follows OptiC's reference-defined 200 bins. As the
original convention discards out-of-range generated mass before normalizing,
the evaluator also reports the discarded fraction and TVD with explicit tail
bins. Mean log p and its absolute error are always reported alongside coverage;
a high mean density alone can reward mode collapse.

## Running

Use the existing `requirements-no-anchor` environment; there are no additional
dependencies. The runner loads the normal OptiQ `.env` through
`optiq_dime.runtime.load_environment`. W&B is online, and initialization failure
aborts the run. No credentials are copied from the source repository.

```bash
# One run, on an explicitly chosen GPU:
CUDA_VISIBLE_DEVICES=0 scripts/run_gmm40.sh --seed 0

# Default 0/1/2 queue on free GPUs 0 and 1:
# Install scripts/supervisor_gmm40.conf.example under supervisor's conf.d,
# then reread/update and start the optiq-gmm40 group.
# Worker 0 runs seeds 0 then 2; worker 1 runs seed 1.
```

Each run stores its exact configuration, source snapshot/hashes, history,
5,000-step checkpoints, final checkpoint, and direct `samples_final.npy` under
`outputs/gmm40/runs`. W&B receives final samples, checkpoint and source artifacts.
Interruptions save an interrupted or stopped checkpoint and preserve history.

For a new checkout, obtain the four released sample files from the official
link above, or copy the existing OptiC downloads. Generated/released research
data stay outside Git:

```bash
mkdir -p outputs/gmm40/reference
cp /workspace/OptiC/DiKL/results/official_drive/GMM40/{dikl,idem,fab,rkl}_samples.pt outputs/gmm40/reference/

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  /workspace/.venv-optiq-no-anchor/bin/python -m benchmarks.gmm40.evaluate \
  --output-dir outputs/reports/gmm40_final
```

The evaluator requires exactly one completed 30k run for each of seeds 0/1/2.
It rejects missing/duplicate runs rather than choosing favorable checkpoints.
Use `--runs-root` to point to a specific experiment's run directory after
running additional configurations. `--baselines-only` permits an independent
released-sample check before training completes.

Validation: `python -m pytest tests/test_gmm40.py -q` checks density parity,
actual actor updates without observations, oracle immutability, metric parity,
exact transport and histogram-tail handling. The repository's existing pytest
fixture records the validation run in W&B.
