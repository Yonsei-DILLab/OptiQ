# Complete DACER-off GMM40 N/M comparison

The user requested N=M64,128,256,512, each with four training seeds and 100k
updates, followed by final visualizations at alpha .2 and 1.05 pt² per point.
The fixed-Q GMM40 iBOLT/TRG adapter has no DACER behavior-noise or learned
entropy controller. The completed 4090 N=M128/256/512 source c429fbb already
implements DACER off. Repeating its 12 identical training jobs would add no
new experimental condition. Train only the missing N=M64 seed0..3 on vast1
RTX4090 using the same learner, target, network, batch, optimizer, temperature,
random latent, mean/sigma setup, evaluation schedule, samples and optimizer
budget. The new plan records `dacer_enabled=false` explicitly.

Before launch, commit/push this plan and controller extension; freeze its full
SHA on vast1. Validate byte identity of the fixed-Q learner and evaluation
functions against the previous N=M64 control. Respect GPU leases and active
unrelated jobs. The registered supervisor controller backfills independently
without retries after a failure. Save and verify final actor count, full-policy
and mu-only raw samples and metrics, W&B runs in OptiQ/gmm-trg, and the new
campaign result. Preserve the previously completed 12 jobs unchanged.

For the final comparison, copy 100k source configs, audits, metrics and sample
arrays for all 16 jobs locally. Check N/M, seeds, 100k actor/optimizer counts,
target identity, shared hyperparameters, sampling mode and raw array shape;
calculate 4-seed mean/sample SD for MMD and mode coverage. Plot all 10k full
policy samples per panel with alpha .2 and marker area 1.05 pt², blue points
over the same GT density contours. Export PNG, PDF, source provenance and a
Korean results note. Retain mu-only as a labeled supplement. Do not describe
this as a DACER-on/off ablation: DACER-on has not been implemented or run on
this fixed-Q benchmark.

Register after committing:

```
python -m gmm40.nm4090_campaign register \
  --plan /home/heechan/OptiQ-ops/sources/<commit>/gmm40/nm4090_dacer_off_64_plan.json \
  --root /home/heechan/optiq-experiments/gmm40-ibolt-dacer-off-nm64-100k-4seed-4090-20260925
```
