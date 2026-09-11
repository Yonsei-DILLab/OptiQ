# OptiQ v2

**Final version: continuous-latent checked K64**, selected on 2026-09-11.
A one-forward semi-implicit tanh-Gaussian actor, scalar twin-Q soft backup with
IDAC entropy estimates, conditional-mixture proposals, full OT Gaussian NLL,
and a sampled soft-score acceptance filter. Training uses T=0.1, beta=1,
16×64 OT, M16 entropy, 256×3 networks, no anchors or extra uniform exploration.

- **[Detailed implementation pseudocode](docs/v2/PSEUDOCODE.md)**
- [Algorithm in Korean](docs/v2/ALGORITHM_KO.md)
- [Installation, execution and independent replication](docs/v2/REPRODUCIBILITY.md)
- [Review and validation](docs/v2/REVIEW.md)
- [Completed Humanoid results](docs/v2/RESULTS_KO.md)
- [Theoretical conditions and limits](docs/v2/THEORY.md)
- [Archived variants and compatibility paths](docs/archive/v2/README.md)

## Run the final version

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

The completed four-seed 900K–1M mean was **5362.67**, versus 5051.65 for the
matched OptiQ control and 5676.49 for the historical 10%-exploration reference.
Fresh final-policy means were **5397.09**, 5224.07 and 5633.18 respectively.
These are whole-algorithm comparisons. OT NLL and the sampled replay-average
filter do not by themselves certify policy improvement.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/v2/final.yaml` | Canonical final numerical configuration |
| `docs/v2/` | Final pseudocode, NumPy reference, reproduction, review and provenance |
| `optiq_dime/` | Shared policy, critic-update, density, OT, NLL and guard implementation |
| `scripts/run_v2.sh` | Default final launcher |
| `configs/archive/v2/` | Original, conditional, guarded, finite and proximal variants |
| `docs/archive/v2/` | Original pseudocode and historical investigations |
| `tests/test_v2_final.py` | Frozen-config, reference-math and common-loop checks |
| `outputs/` | Existing local results/checkpoints; unchanged, ignored by Git |

Historical config names remain small aliases for existing services and reports.
To reproduce the supplied original v2, use
`OPTIQ_CONFIG=archive/v2/original scripts/run_v2.sh 0` explicitly.
Legacy diagnostic/monitor scripts retain paths referenced by saved manifests.

The original DIME, scalar MuJoCo and no-anchor experiment documentation remains
in the [pre-final README archive](docs/archive/README_PRE_V2_FINAL.md).
The upstream project is [DIME](https://arxiv.org/pdf/2502.02316).

## Acknowledgements

Portions are adapted from [UnderdampedDiffusionBridges](https://github.com/DenisBless/UnderdampedDiffusionBridges),
[CrossQ](https://github.com/adityab/CrossQ) and [Stable Baselines Jax](https://github.com/araffin/sbx/).
IDAC supplies the semi-implicit entropy estimator; the OT actor and sampled guard
are this repository's combination. Preserve the repository's licenses and attribution.
