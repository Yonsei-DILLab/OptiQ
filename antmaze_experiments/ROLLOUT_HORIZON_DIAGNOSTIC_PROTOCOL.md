# Paired inference diagnostic of incomplete route acquisition

The completed v3 normalized-geodesic T3/gamma.999 258304-step checkpoint has
16 left entries out of100 direct-policy episodes and no left successes. All16
reach the original700-step limit, at mean final goal distance4.07m;14 move
closer during their last100steps. This does not prove extra time would succeed.

Use that immutable checkpoint, training source
db4ca0a446f5fe8df1dda02e261fa9154c66d9c9, for CPU-only inference. Compare100
episodes at native700 and supplementary1400steps, original identical full
origin/pose/velocity, same per-batch random seed and sampler. Use direct policy
sampling with fresh latent and conditional sigma, without external DACER noise.
Record every failure, goal, route, length, return and raw trajectory. Verify all
paired trajectory prefixes are exactly identical up to each native episode's
termination, and all successes occurring within700 agree. Check checkpoint and
model/optimizer hashes, RNG restoration, original full starts, and reward sums.

Only the reporting process's Gym TimeLimit is changed. Do not alter training,
frozen source, physics, goals, success radius, reward, policy or algorithm. No
restart, continuation learning or changed training setting is authorized by
this diagnostic. The two conditions have different evaluation time budgets;
1400-step successes cannot be counted as native benchmark successes or used to
declare the active goal achieved. A result may motivate a separately specified
experiment, or reveal that extra time does not help. Distinguish this paired CPU
reference from the original GPU evaluation random stream.

Commit/push/freeze the separate reporting source before inference. Run as a
single nice19 CPU supervisor job on180 with4CPU affinity, no GPU allocation,
no autorestart and no4090 use. Preserve the live four training jobs and all
old results. Save results in a separate campaign, archive locally and visually
inspect the paired trajectories. Sharing with199 remains pending SSH recovery.
