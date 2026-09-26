# Fill eligible idle PointMaze GPU slots (2026-09-26)

The user asked to run existing pending jobs in empty GPU slots. This plan
transfers15 previously authorized baseline jobs, not new experiments. Preserve
all running corrected DIPO and baseline workers. MFPO/MEOW multiseed remains held.
All learning settings and frozen source bfb06944705685a7dea84558aba40500c22349b9
remain unchanged:256env,batch4096,16updates/256,8192warmup,1,000,192steps,
62,000updates. DIPO's new32/2048 ratio does not apply to these other baselines.

Under each original46/199 queue.lock, assert each chosen job is still pending,
record its original configuration and mark it transferred before destination
workers can claim it. If any chosen job started, stop the transfer and replan;
never duplicate it or stop its learner. Save transfer sidecars at both origins
and destination with source/controller/plan commits. Preserve cancelled jobs.

Destination campaign pointmaze-baselines-idle-backfill-20260926:
vast-heechan-6 takes6 jobs, initially on idleGPU1/2; guards on0/3 wait for the
current correctedSimple/4-Way DIPO, then independently backfill.
vast-heechan-180 takes9 jobs on healthyGPU0/2/3. GPU1 has a driver-reported
Unknown Error and must remain excluded. Healthy GPUs passed separate PyTorch
and JAX compute checks while holding their existing locks. No driver changes.

The guard now queries only its chosen GPU instead of every device, so the
unrelated failedGPU1 does not abort healthy-device inventory. Once the lock
and idle checks pass, pass CUDA_VISIBLE_DEVICES by UUID to avoid remapping
physical indexes after a device failure. Preserve the20-second idle check,
flock and compute-PID checks; never bypass ownership checks.

New controller adds host180 support but runs each job from unchanged frozen
learner bfb069. Verify8448step/16update preflight separately before main launch;
retain200k/200episode and final500episode evaluation/checkpoint/figure pipeline.
Original shards can contain transferred jobs; aggregate each logical job once
across original and destination roots.15 transferred+remaining9=24 total
baseline multiseed additions. Failure preserves live workers and pauses its
destination pending queue; no automatic retries.
