# iBOLT GMM40 N=M ablation on RTX 4090

User approval, 2026-09-25: N=M in {128, 256, 512}, four fresh training
seeds 0..3 each, 100,000 actor optimizer updates. Only vast1 / RTX 4090 is
eligible. This request supersedes the historical N=M=64 search restriction.

The control is the 100k, 256x3 tuned result used in the current paper figure:
seed0 e0f1ba03e1b7c67eae76b098dcd3c8c34ba14da1 and seeds1..3
29e6fc98b278a41b4f3b365feddd0e931c808f4c. Only N and M change. Retain
256x3 GELU, batch256, Adam3e-4, T1, density beta1, fresh Gaussian latent,
mean-head variance scale16, log sigma[-5,-3.5], initial log sigma -4,
teacher-only standard deviation floor .05, and the same bounded GMM40 target.
No clipping, loss, sampler, initialization or learning-rate changes.

Each update has batch256 independent clouds; each cloud has N latent
components and M teacher actions. Equal optimizer update budgets are NOT equal
Q-query or compute budgets: Q queries are 100000*256*M per run. Model size
remains 133636 parameters for all three values of N/M.

Preserve the original evaluation/checkpoint schedule: 0,100,500,1000,2500,5000,
then every10k through100k. Save 10,000 full-policy samples and 10,000 mu-only
samples separately at each point. Full policy is primary for this experiment,
following the user's latest request to include conditional sigma; retain old
per-run plotting code as historical output and generate clearly labeled campaign
full-policy and mu-only reports separately. MMD is sqrt(max(unbiased MMD2,0));
the MMD2 estimator uses the unchanged first2048 samples and five RBF bandwidths.
Coverage retains the existing component3sigma/minimum-occupancy rule.

Before launch, commit and push all protocol/plan/controller files. Verify that
the TRG adapter, inherited advance/sample/save code, policy, proposal, direct
NLL, box sampler, target and metrics match the reference source byte for byte.
GPU preflight uses the actual batch and N/M, checks finite gradients/parameters,
bounded full/mu samples, optimizer step counts, and checkpoint round-trip. It
also measures steady update time. Preflight models never become training runs.

One independent queue handles all12 runs, filling any eligible idle GPU every
two seconds. Respect the ops GPU lease and BOTH legacy GMM-TRG/NM worker leases,
as well as nvidia-smi occupancy. Do not stop unrelated workers, reuse occupied
GPUs, launch on5090, or restart a failed learner. On failure, hold pending jobs
and let other live jobs finish. The current free GPU can start while other
GPUs remain occupied. Jobs are ordered by seed then128/256/512.

W&B: OptiQ/gmm-trg; group is the campaign name. Runtime authentication is read
from the existing server credential location and never placed in Git or logs.
Exact source SHA, runtime packages, GPU model, job args, preflight proofs and
100k optimizer-count audits are retained. Final reports group by N/M and use
four training-seed mean +/- sample SD; never pool distinct N/M values as seeds.

Register with the committed frozen source using:

```
python -m gmm40.nm4090_campaign register --root /home/heechan/optiq-experiments/gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925
```

The supervisor service manages the controller with automatic restart disabled.
Final config, checkpoints, both sample types, audits, metrics and reports are
preserved in the campaign and packaged with a SHA256 digest after all12 finish.
