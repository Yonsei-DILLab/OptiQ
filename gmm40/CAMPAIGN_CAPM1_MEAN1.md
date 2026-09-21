# GMM40 TRG cap -1, Xavier mean initialization (2026-09-21)

User-approved control: OptiQ Direct GMM/TRG only, seeds 0..3, each 100,000
actor updates. Log sigma bounds [-5,-1], initial log sigma -1 (upper bound),
mean-head variance scale 1.0 (Xavier uniform). Plan: campaign_capm1_mean1_plan.json.
All other settings match the capm3 mean1 campaign: 256x2 GELU, random latent,
N=M64, batch256, T1, beta1 density correction, Adam3e-4, exact box Gaussian
sampling, teacher floor exp(-5), 10,000 full-policy evaluation samples.
RL defaults and previous frozen campaigns are unchanged.

Commit the plan/protocol before training. Freeze the commit in a detached
worktree under /home/heechan/OptiQ-ops/sources/<commit>. Prepare using
python -m gmm40.campaign prepare --root <root> --plan gmm40/campaign_capm1_mean1_plan.json.
Run gmm40.campaign preflight at the frozen source on an idle GPU through
OptiQ-ops/run-gpu.sh (two disposable validation updates). Only after it passes,
register gmm40.campaign run under the existing user supervisor. Retain source SHA,
preflight, manifest and registration. Existing GPU locks and occupancy checks
apply; fill idle slots every five seconds. Do not interrupt other jobs.

Campaign root /home/heechan/optiq-experiments/gmm40-trg-capm1-mean1-100k-4seed-20260921.
Supervisor and W&B group use the same name; W&B project OptiQ/gmm-trg.
The committed wrapper disables argparse abbreviation, fixing the prior W&B
name/method parsing issue. Verify true names/config and saved evaluations.
Failures block pending work and preserve active jobs for review, without
unrequested restarts. Completion requires all four 100k optimizer-count audits;
the existing reporter saves learning curves, final distributions, seed mean/SD,
CSV/JSON, report and SHA256-addressed final archive. Collect locally under
artifacts/gmm40_capm1_mean1_queue.

Against original 87d5d8f, the training change is mean scale 1e-4 -> 1.
Against capm3 mean1, upper and initial log sigma change together from -3 to -1;
do not describe this as isolating only the upper bound.
