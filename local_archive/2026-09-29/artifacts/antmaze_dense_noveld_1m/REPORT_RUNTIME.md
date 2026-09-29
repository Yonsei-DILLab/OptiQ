# Reporting runtime

Use `/tmp/optiq-antmaze-report-20260922/bin/python` for collection with archive verification and for reporting.
This is an isolated Python 3.12 environment; NumPy 2.3.5, Torch 2.14.0, Matplotlib 3.11.2.
The existing system Python 3.14 + NumPy 1.26 is unsuitable for these array audits.
Training source, reporter source, training processes, and saved data were not changed.

The v1 OptiQ 1M archive passed the full unchanged verifier under the compatible runtime.
See `runs/v1-optiq-s0/archive-verification.json` for the actual file hashes, checkpoint digest,
1,000,000 verified replay rewards, and runtime provenance.

Commands from the repository root:

```
/tmp/optiq-antmaze-report-20260922/bin/python -m antmaze.multimodal.collect_dense_noveld --output artifacts/antmaze_dense_noveld_1m --archive-completed
/tmp/optiq-antmaze-report-20260922/bin/python -m antmaze.multimodal.dense_noveld_report --root artifacts/antmaze_dense_noveld_1m
```

The final report requires all 16 completed jobs. Use `--partial` only for explicitly interim reporting.
