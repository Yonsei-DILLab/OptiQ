# v1 same-state supplementary inference

Active goal evidence check only; no training or algorithm change. Preserve
v1's primary random-start evaluation. Historical official v1 dense/T3/DACER-off
source5baa5b3463416cdfdcad465e8f862f3729802568 reaches both routes under random
starts, but the saved fixed-start sample at approximately(1.987,-1.452) takes
only the lower route. Test whether the apparent diversity is entirely driven
by initial position, without replacing the primary evaluation.

Run reevaluate_dense_random with --family official --task v1
--v1-origin-supplement --evaluation-seed20260925 --episodes100 --batch50.
The explicit supplementary flag evaluates identical original pose/velocity
with x=y=0 inside the original training start distribution. Use separately
seeded direct-policy and mu-only evaluation, original frozen sampler/parameters,
no external behavior noise, no updates, no goal-conditioned/best-of sampling.
Report modes separately and keep failed episodes. The full initial state must
be bitwise identical across episodes and paired modes. Recompute goal endpoints
and dense reward sums, verify checkpoint hash and exact restored parameters,
and verify model/optimizer/checkpoint unchanged after inference.

Start with the final3.008M checkpoint. If both successful routes appear, inspect
earlier saved checkpoints and independent evaluation RNG before claiming
retention. A single chosen state does not establish diversity everywhere.
Optional --checkpoint selects an immutable policy checkpoint within the same
run and verifies its own proof and actual step, never labels it as final.

Use CPU-only JAX on four CPU cores under a low-priority supervisor service on180.
Do not consume GPU slots or modify/cancel active training. Commit/push the
reporting source before execution, and record it separately from5baa5b3.
No4090 use. Source-only199 sharing waits for SSH recovery. Primary random-start
datasets and all historical data remain unchanged.

Post-hoc validator correction: the2.75M native probe hit one discrepancy in
the old `stored_distance <= .50002` check. That check expands the physical
success radius and can classify a near miss as success. Preserve its failed
attempt, and repeat only this inference probe under a new reporting commit.
Record the minimum float64 physics distance at every actual transition and
check success against the unchanged exact radius .5. Compare stored float32
coordinates with those distances separately, logging boundary cases; do not
change environment success, rewards, actions or model parameters. Other
already-verified probes retain their original reporting-source provenance.
