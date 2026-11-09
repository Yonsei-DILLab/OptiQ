# MFPO PointMaze entropy grid

Private heechan-rebuttal work only. Parent/root AGENTS.md rules are mandatory;
never export this folder, settings or results to any anonymous branch/snapshot.

Medium and Hard each run a full Cartesian grid:

- target_entropy_coeff: -1.0, -0.5, -0.25
- initial temp: 0.01, 0.1, 1.0 (learned entropy coefficient initialization)
- selection seeds: 5,6; nine configurations × two seeds × two mazes =36 runs
- after selection per maze: five fresh runs, seeds0..4; up to10 further runs

Common settings are the existing PointMaze matched profile: 1,000,192 transitions
including8192 random warmup; vector environments256; batch256; UTD1;992000
updates; replay1M. Actor/critic/log-density/temperature Adam learning rates all
remain3e-4. Networks remain native256×3, layer normalization, T2, policy/Gaussian
proposals16/32, divergence samples2, distributional support[-1600,1600],101 atoms.
The native pinned MFPO vendor and baseline adapter bytes remain unchanged.
Only target entropy and the initial temperature scalar are overridden; optimizer
initial states and the actor/critic/log-density RNG/initialization are retained.

Primary evaluation uses the author's fixed10-candidate target-Q selection;
secondary evaluation uses one direct sample for comparison to previous PointMaze
results. Both modes are clearly labeled, saved at every200K checkpoint, and
evaluated with the same environment/RNG seeds. Both normal and obstacle settings
are retained:200 episodes per intermediate condition/mode,500 at the end.
No learning or training-RNG changes occur during evaluation. Candidate selection
has a different inference cost from a single sample; do not mix their timings.

Selection waits for all18 grid runs for a maze to finish. A candidate requires
both selection seeds to complete successfully; failed/incomplete pairs are
explicitly excluded and their failure records retained. Rank by the mean
official normal success over checkpoints600064,800000,1000192 and both seeds5/6;
ties use the worst-seed late mean, then goal coverage, full evaluation-curve
success and proximity to the default setting. Some observed late success is
required to promote; all-zero outcomes are marked for review. Selection and final
training seeds are disjoint. The selected pair automatically launches seeds0..4
at the same complete budget and evaluation protocol. Final metrics use the seed
mean and sample SD, not pooled episodes or a favorable checkpoint.

Each host stores plan.json, ready.json, selection.json, metrics.json, source hashes,
resolved run configurations, diagnostics.jsonl, evaluation trajectories, full
Flax checkpoints and final replay. Source snapshots are independent of existing
Fetch, PointMaze and reporting controllers. Supervisor manages the new controller;
admission respects actual GPU and host memory and uses dedicated physical cores.
Persistent JAX compilation caching affects startup only and is recorded in the
controller environment. Shared-GPU wall time is not an isolated speed benchmark.

Smoke validation uses the real batch256 and native model/proposal counts with a
short1024-transition buffer. Its four optimizer updates and trajectories are
excluded from all reported training runs.
