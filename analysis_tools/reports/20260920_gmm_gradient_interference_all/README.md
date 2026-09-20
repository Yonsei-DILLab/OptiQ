# Direct GMM gradient interference: final32-run analysis

Read-only analysis of three immutable campaigns,8sizes×4seeds, all20K.
No actor/critic update is run by these scripts. Data/checkpoints/plots are not committed.

Python dependencies: numpy, matplotlib, markdown. The server's existing
`/home/heejoonorm/.venvs/optiq-monge/bin/python` provides these.

```bash
python build_report.py \
  --data /home/heejoonorm/OptiQ/legacy_monge/gradient_interference \
  --out /home/heejoonorm/OptiQ/legacy_monge/gradient_analysis/REPORT_SHA/report \
  --analysis-commit FULL_REPORT_SHA
```

The data directory contains source prefixes1b1c6947d06b,f3aebce92a3e,32bc71599d06.
`analyze.py` can run with numpy only to validate data and export tables.
`build_report.py` creates histogram and intervention/latent plots, a self-contained
HTML, LaTex-preserving Markdown, input hashes and per-run data.

A basin is fixed at[-1,-.3),[-.3,.3),[.3,1]. Specialist probability>=.8.
TV<.1 is a descriptive post-hoc threshold, not a preregistered endpoint.
A positive held-out NLL change>1e-6 is counted for the per-mode harm table;
6 off-diagonal entries per seed are not treated as independent seeds.
Figure averages use equal seed weight. Missing specialist groups are masked,
not replaced with zero. The component plots use identical fixed evaluation z;
training z is freshly sampled and has no permanent row identity.

Exact target curves are analytic; all actor densities use32768 action histograms,
without KDE. Gaussian CDF is used for target bin/basin masses and conditional
specialist membership only. Reference NLL uses the original1024-bin quadrature.
All diagnostic branch data are pre-existing, discarded clone updates.
