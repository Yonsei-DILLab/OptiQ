# GMM40 TRG cap -3, Xavier mean initialization (2026-09-21)

User-approved standalone experiment on vast-heechan-180: OptiQ Direct GMM/TRG
only, seeds 0..3, 100,000 actor updates each. Log sigma bounds are [-5,-3],
initial log sigma is -3 (the upper bound, following the user's initialization
preference), and mean-head variance scale is 1.0 (Xavier uniform).

The explicit plan is `campaign_capm3_plan.json`. The general GMM40 defaults
remain [-5,-1], initial -1, mean initialization 1.0; RL defaults are unchanged.
All other settings are retained: 256x2 GELU, random latent, N=M64, batch256,
T1, beta1 density correction, Adam3e-4, exact box Gaussian sampling and
teacher floor exp(-5). Reference target samples enter evaluation only.
Each evaluation saves 10,000 full-policy samples, the mu-only samples,
coverage, near fraction, MMD, mass TV, configs and optimizer/RNG checkpoints.

The new queue uses any idle GPU immediately and fills newly freed slots every
5 seconds, one run per GPU using the existing GPU locks. It does not stop,
restart or change the original GMM40 or DACER campaigns. There are four fresh
runs only; no extra baselines or mean-only control are launched by this plan.
Failures preserve other active runs, block pending launches and require review.

Commit all sources and this plan first, then create a detached frozen source.
Run `python -m gmm40.validate_capm3` (no learner updates), prepare the campaign
with `python -m gmm40.campaign prepare --root <root> --plan
gmm40/campaign_capm3_plan.json`, and run the committed two-update GPU preflight
before registering `gmm40.campaign run` with supervisor. Preserve the full
source SHA and dependency pins in manifest.json.

Campaign root and W&B group:
`gmm40-trg-capm3-mean1-100k-4seed-20260921` under
`/home/heechan/optiq-experiments`, project `OptiQ/gmm-trg`.
Completed runs must pass the 100k optimizer-count audit. The existing campaign
reporter saves four-seed mean/sample-SD curves, final distributions, CSV/JSON,
a Korean report and a SHA256-addressed portable final-results archive.

Compare with the original frozen `87d5d8f` results only with both changes
disclosed: mean initialization 1e-4 -> 1 and upper/initial log sigma -1 -> -3.
This comparison cannot isolate either mean initialization or the sigma cap.

Logging correction after registration: the original campaign wrapper accepted
abbreviated CLI options, misreading `--n`/`--m` as `--name`/`--method` before
calling the fully specified training parser. Training configs and local results
are correct, but W&B names/configs/history need repair. The wrapper now disables
abbreviations for future sources. The launched frozen source remains unchanged.
`repair_campaign_wandb.py --root <root>` waits for each run to finish, resumes
its original W&B ID, restores the true config/name and all saved evaluations,
uploads final artifacts and verifies the remote summary. It records its own
post-launch source commit in a separate sidecar. It never restarts a learner.
