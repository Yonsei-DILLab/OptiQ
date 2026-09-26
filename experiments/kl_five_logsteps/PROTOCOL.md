# Initialization, 10, 100, 1K, 10K, 100K updates

User request 2026-09-26: show these six times for all five existing environments.
Keep the active 100K campaign and its source snapshots unchanged. Its arrays are
2338457 and 2338506; source commit 62e6da3656a572e4bb093084884b5e0c35ee4e7f.

## Short early replay

Same 40 case/method/seed configurations in kl_five_progress/TASKS.json. Same
N=M=128, batch=32, temperature=.25, actor, optimizer and target functions. Reverse
L=2^20, density chunk4096. Stop after 1K (no duplicate 100K training). Require
original initial parameter hashes; record original and current commits.
Reuse the unchanged numerical kernels from kl_six_highL_1d. Reverse uses the same
5-step compiled blocks. Forward's first 100-step block is split into 10+90 solely
to observe update10; subsequent blocks are100. This does not change minibatch
size or update equations, but compilation/roundoff can differ. Do not claim
bitwise trajectory identity. No ground-truth samples are used for training.

At steps0,10,100,1000 save full checkpoints and 2^20 actual action samples/seed,
512-bin histograms and existing diagnostics. Evaluation must leave serialized
parameters, optimizer and training PRNG byte-identical. Seed range0–3. All cases
and seeds are retained; do not select favorable reconstructions.

## Figure and interpretation

One 2×6 wide figure per environment: columns Initialization,10,100,1K,10K,100K;
rows Forward/Reverse. Also export6×2 variants. Preserve sans-serif style, blue
Forward/orange Reverse, pale fill, thin seed curves and black dashed true target.
Fixed y-axis range over time per environment/method; bottom-only a labels.

Early0–1K comes from the short replay, later10K/100K from the unchanged main
campaign. Explicitly label this segment provenance in the figure/caption/report;
do not present the composite as recovered checkpoints of one byte-identical run.
Compare the two1K snapshots for each seed: histogram-to-histogram TV and fixed
action/mean/sigma probe differences. Report all differences and visibly flag
TV>0.02; this is a display warning, not a seed-selection rule or statistical test.
No interpolated or fabricated densities. A four-column early preview can be
generated immediately after the short jobs; the six-column figure waits for all
main and early runs. Use2^20 samples/seed throughout both segments.

## Execution and storage

Commit before launch, immutable source manifest+full SHA+job IDs. Use spare
SLURM GPUs with at most8 short runs concurrent, one GPU/twoCPUs each; exclude the
known problematic nodes and retain all ongoing jobs. Hardware-only retries keep
the original source commit. Submit separate CPU plotting jobs for early-only and
full results, with result-availability dependencies only. The existing long-run
figures remain registered and unchanged.

New root: extensions/kl_five_progress_20260926/early_steps under the established
login4 scratch path. Primary storage: dildata:/data1/heejoonorm/OptiQ/studies/
20260926_kl_five_progress/campaign/early_steps. Reuse the restricted backup route,
and require both original ten_k and the new full log_steps reports before the
backup loop declares completion. Checkpoints/samples/credentials stay out of Git.
