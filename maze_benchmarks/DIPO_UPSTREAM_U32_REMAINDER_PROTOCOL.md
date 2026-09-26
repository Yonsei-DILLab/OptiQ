# Replace remaining pre-fix DIPO runs (2026-09-26)

The user requested stopping failed pre-fix DIPO and replacing it with the
corrected implementation. Old PointMaze Simple/Medium/Hard are at 800768
transitions with latest success 0/200. Old 8/12/16-Way reach one goal only;
their success is 200/200, so distinguish success from failed multimodal coverage.
Old 4-Way already completed 1,001,472 transitions with 1024/1024 on one goal.
Stop the six still-running old seed-0 jobs, preserving their partial data,
and retain the completed old 4-Way result. Cancel the old campaign's pending
queue; never resume it. The earlier six seeds1-3 were already cancelled.

New Medium/Hard seeds0-2 are already running from source
041738202b5e6bcf765a9a807e917afd2778608a in
pointmaze-dipo-upstream-u32-s0to2-1m-20260926. Keep them and do not duplicate.
DIPO_UPSTREAM_U32_REMAINDER_PLAN.json replaces only Simple and4/8/12/16-Way,
one fresh seed0 each, in maze-dipo-upstream-u32-s0-1m-20260926.
Host6 GPU0/3: Simple/4-Way; host46 GPU1/3:8/12-Way; host199 GPU3:16-Way.
Other baseline GPU workers remain active; temporarily stop only waiting guards
on replacement slots and restore them after the new DIPO workers hold locks.

Use the same corrected learner and UTD as POINTMAZE_DIPO_UPSTREAM_U32_PROTOCOL.md:
2048env,batch4096,32updates/2048 transitions,8192warmup,1,001,472 total
transitions,15,520 learner updates, literal upstream DiffusionMemory.replace.
Keep all task-native rewards/dynamics and DIPO models/optimizer settings.
Simple uses original DrAC sparse +100; Way tasks retain their existing symmetric
wall-free rewards. Final500 PointMaze episodes /1024 Way episodes. Every200k
evaluate200 episodes and save raw trajectories/checkpoints/automatic figures.
Every job independently passes10,240steps/32updates preflight; commit and share
exact source/plan before launch, no automatic retries or performance stopping.

This plan-only follow-up does not change the learning code relative to0417382.
Record its distinct frozen commit anyway. Track all corrected11 jobs (six
Medium/Hard plus five replacements); aggregate only compatible configurations.
Preserve old source4b46/results as historical writeback/UTD128 comparisons.
