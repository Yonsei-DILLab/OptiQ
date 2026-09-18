# Numerical diagnosis artifacts

These compact tables, figures and verification records support
[RESIDUAL_DIAGNOSIS_KO.md](../../RESIDUAL_DIAGNOSIS_KO.md). They describe the
earlier mean-OT checkpoints with Sinkhorn epsilon=.25 and gradient clip=2,
not the later epsilon=.1/.05 no-clip training grid.

| Files | Scope |
|---|---|
| `benchmark_*` | Matched-width historical learning curves and small-sample uncertainty |
| `mechanism_diagnostics.png` / `.pdf` | Exportable figure of frozen-policy and teacher diagnostics |
| `proposal_bias.csv`, `projection_geometry_score.csv` | Finite proposal coverage and projection geometry |
| `counterfactual_*`, `independent_mc_combined.csv` | Paired first-action rollout comparisons and independent repeats |
| `trajectory_error_*` | Pathwise Bellman/twin-min/target-lag decomposition |
| `frozen_objectives_*`, `early_objectives_*` | Bounded actor fits with a fixed critic and separate evaluations |
| `deterministic_oracle_*` | Zero-z continuation diagnostics and their numerical/causal limitations |
| `completed_campaign_*` | Previously authorized uniform/annealing cohorts, completed at 1M |

The report distinguishes measured effects, negative controls and unresolved
causality. Full trajectories, numerical scripts and original checkpoints are
instance artifacts under `/root/anal/v5_residual_cause_20260913` and the cited
experiment directories. They are not embedded in this compact documentation
bundle. Verification JSONs retain input paths/hashes; reproducing those numerical
audits requires the preserved inputs. Portable algorithm tests are in `tests/`.
CSV line endings in this documentation copy are normalized to LF; cell values
are unchanged. Verification hashes refer to the preserved original files.
