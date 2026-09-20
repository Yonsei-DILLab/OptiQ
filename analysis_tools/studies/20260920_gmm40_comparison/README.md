# GMM40 completed-run report

Analysis only; no training, changed hyperparameters, or reruns.
Numerical source: 3f34dbedca707b582f02034e915be6f49fb716a6.
32 runs (8 conditions x seeds0-3), all 75K updates.

Run with Python + numpy/scipy/matplotlib/markdown:

```sh
python build_report.py --workspace /Users/heejoon/Documents/ChatGPT/OptiQ
```

Inputs: `studies/20260920_gmm40_comparison/results/`, `analysis_exports/`,
source manifest, protocol and validation receipt. Raw data/checkpoints stay on
dildata, outside Git. `analysis_exports` contains final teacher b,w,q,log_q,pos,
conditional heads when applicable; seed0 also has effective assignment original
and sorted contiguous block sums (at most128x128), sort indices, original shape,
and row/column masses. Full matrices remain in original run archives.

The builder verifies all32 final sample hashes against EXPORT_MANIFEST, recomputes
saved evaluation metrics, checks all16 evaluation points and five15K segments.
All actor histograms use32,768 raw latent/action-noise draws with no smoothing.
Seed0 is the fixed representative; all32 final runs are shown in an appendix.

Outputs: `reports/20260920_gmm40_comparison/report.md`, offline self-contained
`report.html`, figures and machine-readable aggregates/counts/provenance.
FINAL_BACKUP_VERIFIED.json records checkpoint and final sample hashes on dildata.
Central report: dildata:/data1/heejoonorm/OptiQ/reports/20260920_gmm40_comparison/.
