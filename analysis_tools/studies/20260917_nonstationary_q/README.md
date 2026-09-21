# 1D non-stationary Q: exact OT, Sinkhorn, and Direct GMM

This self-contained snapshot is the source used by the 2026-09-17 login4
campaign. Read [PROTOCOL.md](PROTOCOL.md) for the scientific settings and
[CURRENT_STATE.md](CURRENT_STATE.md) for submission IDs and storage locations.

Seven methods × four N×M settings × four seeds × four trajectory types give
448 final comparisons. Shared 20K prefixes and four learned-Q sources bring
the execution graph to564 nodes. `tasks.json` fixes all method/size/seed pairs.

`v5/` contains the exact v5 actor/critic implementation used by this experiment,
including the Direct GMM loss. It is intentionally preserved independently of
the repository-root historical implementation. Use this study's source when
reproducing these runs. All384 files in `SOURCE_MANIFEST.json` match the source
validated and launched on login4; the four `VALIDATION_*.json` files preserve
the numerical checks. This Git snapshot was committed after the current
campaign had been launched; future launches must record a commit first.

## Reproduction and reports

Stage this directory as a standalone run root and set `PYTHONPATH` to both
that root and its `v5` subdirectory. The validated environment is Python3.11,
NumPy1.26 and JAX0.4.33. Consult the vendored dependency locks and protocol.

The existing `job.sbatch`, `report.sbatch`, and `ops/submit.py` retain their
registered login4 paths and resource limits. Inspect existing submissions
before any resume; do not launch duplicate arrays. If porting to another
location, update those paths deliberately and commit the new configuration.

`build_report.py --root RUN_ROOT --out OUTPUT_DIRECTORY` produces the MD/HTML
report, sampled-action histograms, teacher diagnostics, fixed-latent traces,
and original/action-sorted assignment heatmaps. Checkpoints and raw samples
are on dildata, not in Git. The Git repository retains the small numerical
fixtures already included in the original v5 source manifest.

Central archive: `dildata:/data1/heejoonorm/OptiQ/studies/20260917_nonstationary_q`.
