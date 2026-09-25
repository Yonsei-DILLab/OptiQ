# Collection concurrency with the global update ratio preserved

The active goal permits reward/hyperparameter experiments, keeping the algorithm
structure unchanged. At258304 global transitions with256 environments, each
environment advances1009 times and has only one700-step timeout opportunity.
The two completed v3 teacher-floor controls recorded zero training goal visits
despite successful final-policy evaluation on the right. During a700-step
episode after warmup, the current policy receives5600 optimizer updates.
These facts motivate a collection-cadence test; they do not prove the cause.

Register two fresh seed0 v3/v4 runs on otherwise free180 GPUs. Use32 environments
and one learner update per32 transitions, replacing256 environments/eight
updates per256. Keep the global ratio1/32, batch4096, replay1M, warmup8192,
258304total transitions and7816learner updates identical. Each environment now
advances8072 steps by the end and sees700 rather than5600updates over a full
post-warmup episode. Collection order, replay composition and within-trajectory
policy changes necessarily differ; this is not a claim of identical data or a
pure optimizer test. Fewer parallel environments may increase wall time.

v3 control: d259301, normalized-geodesic progress100, T3, teacherfloor1.
v4 control:555bb7e, original-geodesic progress100, T1, teacherfloor.5.
Both use gamma.999, H/d+.7/500, random normal latent,N=M64, sigma[-5,-1]/init-1,
256x3GELU,meaninit1,Adam actor3e-4/critic5e-4,tau.005,beta1,NovelD OFF.
No finite-prior option, reward, physics, horizon, reset or success changes.
Keep all nine computational files pinned to2564b59. The adapter exposes a
collection-count option and changes only how many existing updates are called
between collection batches; no TD, loss, optimizer, actor or sampling edits.
The absent option retains256/eight and every previous default.

Keep40native and40direct evaluations at exactly the same actual global
transition counts as the256-env controls:50176,100096,150016,200192,250112.
The final258304 evaluation uses100episodes per mode and full checkpoint.
Original v3/v4 full-state origin, native700 horizon. Direct includes conditional
sigma and no external DACER noise. Native is random-z mu-only. Do not conflate
training visits with final-policy successes or first-gate entries with successes.

Commit/push/share/freeze before registration. Each job independently passes a
real8448transition/eight-update/batch4096 GPU preflight and full reward/replay
readback before main. With32env this is eight collection/update blocks, not the
old one block. Verify main and preflight initial actor/critic hashes against
the corresponding control, actual32 simulator states,1/32 counters at every
block, exact evaluation clocks, unchanged native learning config and real GPU
source/PIDs. Preserve live fixed64 runs on199 and every old artifact. No4090,
cancelled resumes, extra seeds, automatic retry or extension. Screen at250k;
both successful routes, followed by later retention, are needed for the goal.
