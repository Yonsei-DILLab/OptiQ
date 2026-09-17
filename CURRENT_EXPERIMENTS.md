# Current OptiQ experiments

## 2026-09-17: non-stationary Q tracking

The current 1D experiment is preserved in
[`analysis_tools/studies/20260917_nonstationary_q/`](analysis_tools/studies/20260917_nonstationary_q/README.md).
Its own `v5/` tree is the validated execution source: Direct GMM, exact OT,
and Sinkhorn with learned or fixed sigma. The snapshot includes all384
source-manifest files,448 comparison trajectories in the task plan, validation
records, Slurm launchers, and the complete MD/HTML report generator.

The experiment uses the study's source and protocol, not the historical
repository-root entry point. Running jobs keep their frozen numerical source.
See [execution records](analysis_tools/studies/20260917_nonstationary_q/CURRENT_STATE.md)
and [settings](analysis_tools/studies/20260917_nonstationary_q/PROTOCOL.md).

From2026-09-17 onward, commit source/configuration before launching new
experiments and record the full commit SHA with each campaign. The current
snapshot was committed after launch and retains the original source hashes.

## 2026-09-11: historical Boltzmann and batch-one experiments

The `heejoon` branch preserves the code used for the Boltzmann-backup study,
including the current batch-one frozen-Q and GMM40 campaign. Model parameters,
evaluation arrays and training logs are archived on lab storage, not added to Git.

## Executed source

The repository root combines the production checkout `mujoco-setting` at
`7e2da67d2f0988f6f211b635f7311d32a0c8e8c6` with the exact source files from
`20260911_batch1_v1/code`. The latter retains the production critic and imports
the GMM-capable actor update and transport from `heechan-no-anchor` at
`1d9394ad7e35d18856115f47983d19f81e4f136e`.

This is the actual experiment implementation, not the newer GitHub
`mujoco-setting` tip `48bd296be12e7adcaa9b6af061eb02d64daf01f8`.
No active cluster checkout or running experiment was switched during publication.
Legacy launchers remain available; use the campaign-specific code and protocol
when reproducing historical results.

| Campaign | Preserved code and configuration |
|---|---|
| Original Boltzmann/operator study | `experiment_snapshots/20260909_v1/` |
| Valid categorical-target ablation | `experiment_snapshots/20260910_categorical_v2/` |
| Batch-one frozen-Q and GMM40 | `experiment_snapshots/20260911_batch1_v1/` |

The invalid categorical v1 data remains in the full storage backup for provenance;
it must not be used as categorical evidence. The original study's active analysis
scope is seeds 0–4; extra historical runs are retained without changing that scope.

The batch-one campaign uses temperature 1, proposal std 1 and five training seeds.
Version 1 uses 16 actor samples and four random candidates plus one anchor per
center. Version 2 uses 2048 actor samples and one random candidate per center.
Frozen-Q uses bounded, truncated Gaussian proposals; GMM40 uses unbounded
Gaussian KDE. The complete settings and remaining architecture differences are
in the campaign README and protocol. Both variants retain row-argmax/MSE targets.

## Integrity and execution

`experiment_snapshots/release_verification.json` records the source-hash check:
all 83 non-OS-metadata files in the active campaign's recorded source manifest
match both the root files and the preserved campaign snapshot. These files were
already exercised by the campaign's CPU reference and GPU actor-update validation.
Publishing this copy does not rerun training or establish new performance results.

Campaign commands and task manifests retain their original login4 absolute paths
for provenance. On another host, stage a campaign into a writable run directory,
use its own `code` directory as `PYTHONPATH`, and update launcher/run paths before
execution. Do not launch the retained Slurm scripts unchanged on Vast.ai.
The original Python dependencies are in `requirements-mujoco.lock`; the storage
backup also records the actual environment package list.

## Storage

Lab storage root: `dildata:/data1/heejoonorm/OptiQ`.

- `backups/20260911/login4/OptiQ/`: full original cluster repository and outputs.
- `backups/20260911/local_workspace/`: local reports, analysis code and saved data.
- `backups/20260911/metadata/`: inventory, transfer and integrity-verification records.
- `repos/OptiQ.git`: Git repository mirror for code recovery.

The backup is a dated snapshot. Jobs running after its capture keep producing new
files on login4; they require a later synchronization. Neither checkpoint pruning
nor changes to the scientific experiment settings are part of this publication.
