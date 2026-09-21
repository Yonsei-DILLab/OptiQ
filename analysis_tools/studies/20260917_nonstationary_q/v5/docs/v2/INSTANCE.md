# Canonical checked-v2 paths on this instance

The successful baseline is continuous-latent **checked K64**, not Gaussian-W2.
Reference training commit: `8cb4f237aacb61113f65e693e23236f17d77f978`.
The imported final release is `894e4aa030ef88da309f1a93912d5921d4db3e5e`.
It centralizes the successful configuration; historical update parity and
release source hashes are checked independently after this local migration.

## Single source of truth

| Purpose | Path |
| --- | --- |
| Active repository / branch | `/root/OptiQ`, `v2` |
| Python environment | `/root/.venv-optiq-mujoco/bin/python` |
| Credentials | `/root/OptiQ/.env` (never copy into source snapshots) |
| Canonical numerical config | `/root/OptiQ/configs/v2/final.yaml` |
| Public config aliases | `mujoco_v2`, `mujoco_v2_checked`, `v2/final` |
| Pseudocode | `/root/OptiQ/docs/v2/PSEUDOCODE.md` |
| Default launcher | `/root/OptiQ/scripts/run_v2.sh` |
| Managed workers | `optiq-v2-checked64:optiq-v2-checked64-0` through `-3` |
| New outputs | `/root/optiq-experiments/v2_checked64/outputs/` |
| Experiment protocol | `/root/optiq-experiments/v2_checked64/PROTOCOL.md` |
| Migration checks and validation | `/root/anal/v2_unification_20260911/` |
| Previous local code archive | `/root/optiq-archives/v2-before-checked64-20260911/` |

There is no active `/workspace/OptiQ-v2` checkout on this instance. That path
in older reports identifies the other machine that ran the successful trials.
Do not make this instance load code or credentials from that old path.

## Baseline and logging

Use `OptiQ/optiq_mujoco_v2_confirmation`, the successful baseline project.
Seed 0 `22qquwe2` finished at 5309.45; seeds 1/2/3 are `7apzqdg7`,
`w9cj6bap`, `y59aoauh`. Keep those historical runs intact.
New runs get unique IDs and timestamp/UUID output directories. The launcher
clears stale W&B resume/run/project variables before composing the config.
It verifies the canonical learning-source hashes and frozen numerical defaults
before launching, so a different local algorithm cannot silently use this entry.

Defaults: scalar twin critics, actor/critic 256x3, T=.1, beta=1, M16/K64,
full OT conditional NLL, sampled soft-score guard, norm-2 gradient clipping,
LN=false, no anchor, no extra uniform collection, no annealing or ESS control.
One critic update and one candidate actor update follow each environment step
after 5K warmup. Failed guard checks restore actor parameters and Adam state.
These checks are not a global true-Q policy-improvement certificate.

## Inspect and run

```bash
cd /root/OptiQ
scripts/run_v2.sh --list
scripts/run_v2.sh 0 --check
/root/.venv-optiq-mujoco/bin/python scripts/verify_v2_final.py
supervisorctl status 'optiq-v2-checked64:*'
# Launch only when a training experiment is requested:
supervisorctl start 'optiq-v2-checked64:*'
```

Workers are installed with autostart=false and autorestart=false. Setup and
validation do not automatically start a new four-seed experiment. Requested
training uses a 1M budget with no performance-based early stopping.

For intentional ablations, use explicit overrides with distinct group/run
names in the same baseline project, record the diff, and label the result as
a variant rather than the unchanged successful algorithm.

## Historical material

Existing `/root/OptiQ/outputs`, `/root/optiq-experiments/*`, W&B histories,
frozen snapshots and old supervisor wrappers remain intact for provenance.
Old Gaussian-W2 services are historical, not the canonical launch route.
Do not restart them to launch checked v2. Existing frozen manifests still
refer to their original files; do not rewrite them to pretend they used v2.

The local pre-migration tracked and untracked source is in `source.tar.gz`;
`tracked-changes.patch` and `repository.bundle` preserve changes and Git
history including the named stash. Outputs and the private `.env` remain
at their original paths. The archive is not a new active checkout.
