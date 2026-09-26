# PointMaze experiment entry points

Branch: `pointmaze`. The vendored upstream environments remain unchanged.
Learning adapters and vectorized collection live in `maze_benchmarks/`.

## Current experiments

- Seven-method controls: `DEADLINE_1M_PLAN.json` and `DEADLINE_1M_PROTOCOL.md`.
  Existing training source: `a55aaf13f7803b5cf8da7ba56f318675c7794171`.
- Additional OptiQ Simple/Medium/Hard × T5/T10, seed0:
  `POINTMAZE_T5_T10_PLAN.json` and `POINTMAZE_T5_T10_PROTOCOL.md`.
  Training source: `bfb06944705685a7dea84558aba40500c22349b9`.
  Reuse the three completed T3 controls; do not launch them again.

PointMaze uses the original DrAC maps, sparse +100 goal reward, 256 environments,
batch4096, 16 updates per256 transitions, warmup8192, and1,000,192 total
transitions (62,000 learner calls). Other method-native settings remain in
`agents.py`. This is distinct from the historical AntMaze dense reward studies.

## OptiQ evaluation contract

Each action uses a newly sampled Gaussian latent z and returns mu(s,z).
Conditional sigma is excluded. This is neither zero-z nor one z held through
an episode. Normal rollouts, obstacle rollouts, action arrows and learned-Q
probes obey the same rule. `run.evaluate` rejects sigma-included OptiQ modes.
Training still calls `act(..., mode="train")` and retains conditional sampling.
SAC, SQL, MEOW, MFPO, DIPO and TD3 keep their native sampling semantics.

Primary figures/tables use `mu_only` for OptiQ. Preserved full-policy data is
supplementary. Older obstacle evaluations and Q probes without mu-only data
must not be relabeled; report those as unavailable in a mu-only comparison.
The removal SR5 boundary correction is recomputed from raw goal IDs and does
not change learning or preserved raw records.

`plot_style.py` sets trajectory width1.8pt/alpha0.5 and learning curves2.4pt.
PointMaze final reports show all500 rollouts, including rare goal visits that
can be absent from a first100 subset. Intermediate reports show all available
rollouts up to500. Goal coverage means observed at least once, not balanced modes.
Reported goal counts always use all episodes, including failures; plotted
subsets are deterministic and are not duplicated to make lines thicker.

## Commands

Run commands and actual source SHAs are recorded by `deadline_queue.py` under
logs/ and manifest.json before a real8448-transition/16-update GPU preflight.
Only after it passes does that job begin main training. GPU guards respect
existing process occupancy and locks. Jobs have no automatic retries.

Local verification and current reporting:

```sh
python -m unittest maze_benchmarks.test_mu_only maze_benchmarks.test_evaluation_metrics
python -m maze_benchmarks.collect_deadline --output ARTIFACTS \
  --historical-root HISTORICAL_NWAY --archive-completed --render --verify-all
python -m maze_benchmarks.collect_pointmaze_temperatures --output TEMPERATURE_ARTIFACTS \
  --controls ARTIFACTS --archive-completed --render
```

Never overwrite frozen training sources to update a report. Reporting-only
commits are separate from the training source recorded in each experiment.

For missing historical mu-only obstacle/Q evaluations, `posthoc_mu.py` restores
a preserved checkpoint, checks its original algorithm configuration, and asserts
zero learner updates and unchanged actor/critic/checkpoint bytes. Outputs go in
a separate directory with both source SHAs; existing raw files are never edited.
The plotted mu-only Q surface is E_z Q(s,mu(s,z)); because the critic was trained
with the training sampler, it is not automatically V of the mu-only rollout.

After post-hoc evaluation finishes, collect its separate proof and arrays with
`python -m maze_benchmarks.collect_posthoc_mu --root ARTIFACTS`. The collector
checks the restored checkpoint/config and preserved mu-only rollout hashes
against the verified training archive, and checks that no learner update ran.
Re-render the main report to include mu-only Q surfaces and final obstacle SR5.
