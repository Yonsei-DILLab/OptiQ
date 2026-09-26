# Five-target learning progression, every 10K updates

User request (2026-09-26): show all five environments every 10K updates. The
current five-target paper plots use reverse KL L=2^20. Do not substitute the
older L=1024 two-target figure. Original sample snapshots exist at 0, 1K, 10K,
25K, 50K, 75K and 100K only; the original checkpoint was overwritten. Missing
states must be obtained by replay, never by interpolation or fabricated data.

## Scope and settings

Five targets, in paper order: t00_reference (three Gaussian modes),
n00_spike_ramp, n07_spike_flat_ramp, t01_two_offset, t05_unequal_mass.
Two methods × five targets × seeds 0–3 = 40 reconstructed trajectories.
Forward has L=0; reverse uses L=1048576 at every update. N=M=128, batch=32,
100K updates, Adam 3e-4, action bound 10, mean-head scale 3, hidden widths
256,256; all target/actor/optimizer settings come from each original RUN.json.
TASKS.json contains these exact settings and original commit/hash provenance.

The inherited numerical kernels are unchanged: kl_mode_missing_search_1d,
kl_forward_wide_1d, kl_diverse_targets_1d, kl_nongmm_targets_1d and
kl_six_highL_1d. Reverse density chunk=4096, train block=5, checkpoint
interval=100. Forward chunk=256, block=100, checkpoint interval=1000. Retain
the original PRNG mode (threefry_partitionable=False), default source-action
matmul precision and highest precision inside the density estimator only.
Do not share or reuse MC banks between the 32 groups or updates.

## Added observations, reproducibility, and comparison

Save full actor, optimizer and training RNG checkpoint at 0,10K,...,100K.
At each 10K point sample 2^20 actual actions, in independent chunks of 16384
using eval seeds 197+104729*i. Compute 512-bin histograms, TV, mode metrics and
1D W1. Also observe original 1K/25K/75K for reproduction diagnostics. Evaluation
must leave the serialized optimizer and training RNG byte-identical.

The original initialization hash must match before a run starts. Compare the
first 128 actions, means and sigmas from the original fixed evaluation draw at
each shared saved point. Use the original draw shape (32768 for intermediate
snapshots; 16384 for the first chunk of final 1M evaluation). Record max absolute
error and RMSE, including any mismatch. Cross-hardware trajectories may differ;
call them reconstructed trajectories, not recovered original checkpoints. Do
not splice original final distributions onto reconstructed intermediate runs.
Do not exclude seeds based on whether they reproduce the expected conclusion.

## Execution and storage

Commit code, launch scripts, this protocol and TASKS.json to heejoon before
launch. Store full SHA and source/input manifests with every campaign and job.
Independent login4 arrays use one GPU and two CPUs per run; use disjoint index
sets across GPU pools, retaining known bad-node exclusions. First validate
both target families and methods, real large-L updates, and exact resume after
sampling on each selected GPU pool. Periodic checkpoints and signal handlers
retain interrupted progress. There is no requirement for one environment to
finish before another starts. A CPU report job runs after the training arrays.

Data/checkpoints stay outside Git. Primary archive is
dildata:/data1/heejoonorm/OptiQ/studies/20260926_kl_five_progress.
Use the existing restricted login4 backup route every three minutes. New
source: extensions/kl_five_progress_20260926 under the established login4
scratch root. Do not touch unrelated held jobs.

## Figures

Provide original saved-time plots immediately, marked as such (32768 samples
per intermediate seed, 2^20 at 100K). Reconstructed plots contain every 10K from
10K to100K and 2^20 samples/seed throughout. Export one 10×2 figure and a 2×10
wide variant per environment. Forward blue, reverse orange, faint fill under
the four-seed mean histogram; thin seed curves; exact target black dashed;
fixed y scale over time within each environment/method, sans-serif fonts,
compact spacing, shared method headings, bottom-only a labels. No KDE. Include
numerical input hashes, metrics CSV and a report page.
