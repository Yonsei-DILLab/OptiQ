# Approved RTX 5090 GMM40 campaign (2026-09-21)

After the existing Humanoid DACER seeds 0..3 finish, run fixed-Q GMM40 for
100,000 actor updates for each of OptiQ Direct GMM/TRG, SAC, DIPO, MEOW, MFPO
and JAX SQL, seeds 0..3 (24 runs). DiKL supplies the original GMM40 target;
it is not another training method. No navigation or new MuJoCo run is included.

`campaign_plan.json` is the committed plan. The frozen worktree commit and all
baseline submodule pins are recorded in the runtime `manifest.json` before
launch. All six adapters must pass a two-update GPU preflight using batch 256.
The controller checks completed predecessor jobs, actual GPU processes and the
existing OptiQ GPU locks. One job occupies each GPU. When it exits, the next job
is eligible immediately; there are no algorithm or seed completion barriers.

Common settings: fixed Q = log p_original(40*a), T=1, batch 256, 10,000 IID
evaluation samples, and 100k actor updates. Evaluation schedule is 0, 100, 500,
1000, 2500, 5000, then every 10k. Reference samples only enter evaluation.
Training seeds are 0..3; the evaluation RNG never advances the learner RNG.

The original frozen campaign at `87d5d8f` uses the `20260920_truncated_mll/optiq_dime` actor, proposal and
direct marginal NLL, with 256x2 GELU layers, fresh random latents, N=M64,
log sigma [-5,-1], initial -1, LR3e-4, beta1 density correction, exact mixture
sampling, no extra teacher sigma floor, no anchors, OT, clipping or DACER.
The fixed-Q adapter is `gmm40/optiq_trg.py`; the old `--method optiq` remains
the historical v5 OT adapter and is not selected by this campaign.

User amendment (2026-09-21): new GMM40 `optiq_trg` runs raise the actor's
log-sigma upper bound from -1 to +1, initialize log sigma at that upper bound
(+1; lower bound -5 retained), and change the
mean-head variance initialization scale from 1e-4 to 1.0. Only these two model
settings and the explicitly requested matching sigma initialization change.
Teacher floor exp(-5), fresh latents, N=M64, batch256, T1,
architecture and optimizer remain as above. Existing frozen runs retain their
original source and settings. The original box Gaussian routines are unchanged;
their normalization, samples and gradients are checked over the widened range.
No changed noise distribution or smooth sigma mapping is introduced.

SAC retains its 256x2 Gaussian actor and LR3e-4. DIPO retains the original
100-step diffusion network, LR3e-4, 20 action-improvement steps and clipping.
MEOW retains its original flow architecture and LR1e-3. MFPO retains its 256x3
velocity/divergence networks, embeddings, two sampler steps and LR3e-4. SQL
uses the JAX port with 256x2 layers, LR3e-4 and 16 SVGD particles (8x8 split).
No claim of identical model capacity or equal Q-query compute is made; actual
parameter counts, query counts and elapsed training times are saved per run.

Queue root: `/home/heechan/optiq-experiments/gmm40-5090-100k-4seed-20260921`.
Supervisor runs `python -m gmm40.campaign run --root <queue-root>` from the
frozen worktree. Runtime is `/home/heechan/.venv-optiq-gmm40/bin/python` on
vast-heechan-180 (PyTorch CUDA 12.8, JAX). The launcher retains existing GPU
locks and credential loading. W&B mirrors evaluations to `OptiQ/gmm-trg`,
group `gmm40-5090-100k-4seed-20260921`; local results remain authoritative.

Each run retains configs, source hashes, optimizer/RNG checkpoints, samples,
metrics, visualizations and an optimizer-count audit. A failed run blocks new
launches without killing other running jobs or silently restarting a seed.
Controller restarts adopt live PID identities and never overwrite run folders.
No performance-based early stopping is enabled.

After all 24 runs pass the 100k update audit, `campaign_report.py` generates
four-seed mean/sample-SD curves, final metric bars, all-seed distribution
panels, JSON/CSV and a Korean report. `final-results.tar.gz` contains these,
the exact manifest, final checkpoints/samples and every evaluation metric.
Intermediate results remain in the campaign's `results/` directory. A local
heartbeat collects the final archive and report under
`artifacts/gmm40_5090_queue/results/`, verifies its SHA256 and reports completion.
