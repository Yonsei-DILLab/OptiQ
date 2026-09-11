# OptiQ v2

**Final version: continuous-latent checked K64**, selected on 2026-09-11.
A one-forward semi-implicit tanh-Gaussian actor, scalar twin-Q soft backup with
IDAC entropy estimates, conditional-mixture proposals, full OT Gaussian NLL,
and a sampled soft-score acceptance filter. Training uses T=0.1, beta=1,
16×64 OT, M16 entropy, 256×3 networks, no anchors or extra uniform exploration.
The successful training source is `8cb4f237aacb61113f65e693e23236f17d77f978`.
This is the checked-K64 baseline, not the previous local Gaussian-W2 variant.
Its defaults include **LayerNorm=false, gradient clipping=2, no annealing and
no ESS-based temperature control**.

- **[Detailed implementation pseudocode](docs/v2/PSEUDOCODE.md)**
- [Algorithm in Korean](docs/v2/ALGORITHM_KO.md)
- [Installation, execution and independent replication](docs/v2/REPRODUCIBILITY.md)
- [Review and validation](docs/v2/REVIEW.md)
- [Completed Humanoid results](docs/v2/RESULTS_KO.md)
- [Theoretical conditions and limits](docs/v2/THEORY.md)
- [Archived variants and compatibility paths](docs/archive/v2/README.md)
- [Canonical paths and launch procedure on this instance](docs/v2/INSTANCE.md)

## Run the final version

Install the pinned environment and configure W&B credentials using the
[reproduction guide](docs/v2/REPRODUCIBILITY.md) first. On the current server,
follow the [instance guide](docs/v2/INSTANCE.md); do not install historical
deployment configs as the default workers.

```bash
scripts/run_v2.sh --list
scripts/run_v2.sh 0 --check
# Training on an allocated GPU, after setting credentials and Python:
CUDA_VISIBLE_DEVICES=0 scripts/run_v2.sh 0 progress_bar=false
```

`mujoco_v2`, `mujoco_v2_checked` and `v2/final` resolve to the same final
configuration defined in [configs/v2/final.yaml](configs/v2/final.yaml).
The direct runner `python run_optiq_dime.py` also defaults to final v2.
Humanoid seeds 0/1/2/3 are the reference; `benchmark=ant`, `halfcheetah`,
`walker2d` or `hopper` selects another supported MuJoCo environment.
The launcher checks the frozen learning-source hashes and numerical defaults
before running. Requested experiments use 1M steps without performance-based
early stopping. The installed four-seed workers do not start automatically.

W&B uses `OptiQ/optiq_mujoco_v2_confirmation`, the successful baseline project.
Keep comparisons in that project and distinguish variants by run/group names.
New results go outside the repository to
`../optiq-experiments/v2_checked64/outputs/`; each run gets a fresh directory and
W&B ID. Credentials, checkpoints, local archives and analysis are not committed.

## Results and validation

The completed four-seed 900K–1M mean was **5362.67**, versus 5051.65 for the
matched OptiQ control and 5676.49 for the historical 10%-exploration reference.
Fresh final-policy means were **5397.09**, 5224.07 and 5633.18 respectively.
These are whole-algorithm comparisons. OT NLL and the sampled replay-average
filter do not by themselves certify policy improvement.

The local path migration passed 181 regression tests, a five-test path-module
rerun, CPU/GPU historical-update parity checks, and a 5100-step Humanoid GPU
smoke including checkpoint restoration. These are code/integration checks,
not a new 1M reproduction. See the [review](docs/v2/REVIEW.md) for their scope.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/v2/final.yaml` | Canonical final numerical configuration |
| `docs/v2/` | Final pseudocode, NumPy reference, reproduction, review and provenance |
| `optiq_dime/` | Shared policy, critic-update, density, OT, NLL and guard implementation |
| `scripts/run_v2.sh` | Default final launcher |
| `deploy/supervisor/README.md` | Canonical managed workers versus historical deployments |
| `configs/archive/v2/` | Original, conditional, guarded, finite and proximal variants |
| `docs/archive/v2/` | Original pseudocode and historical investigations |
| `tests/test_v2_final.py` | Frozen-config, reference-math and common-loop checks |
| `tests/test_v2_instance_paths.py` | Canonical launcher, output and deployment path checks |
| `../optiq-experiments/v2_checked64/outputs/` | New checked-v2 results and checkpoints |
| `outputs/` | Historical local results; preserved, not the new default |

Historical config names remain small aliases for existing services and reports.
To reproduce the supplied original v2, use the direct runner with
`--config-name=archive/v2/original` explicitly. The default launcher accepts
only the three checked-K64 config aliases; historical variants are not defaults.
Legacy diagnostic/monitor scripts retain paths referenced by saved manifests.

The original DIME, scalar MuJoCo and no-anchor experiment documentation remains
in the [pre-final README archive](docs/archive/README_PRE_V2_FINAL.md).
The upstream project is [DIME](https://arxiv.org/pdf/2502.02316).

## Acknowledgements

Portions are adapted from [UnderdampedDiffusionBridges](https://github.com/DenisBless/UnderdampedDiffusionBridges),
[CrossQ](https://github.com/adityab/CrossQ) and [Stable Baselines Jax](https://github.com/araffin/sbx/).
IDAC supplies the semi-implicit entropy estimator; the OT actor and sampled guard
are this repository's combination. Preserve the repository's licenses and attribution.
