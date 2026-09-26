# Way-grid GPU fallback, 2026-09-26

The first 4-Way T=1 preflight on vast-heechan-180 GPU2 failed before any
learner update. A fresh CUDA process cannot initialize even with only GPU2
visible; GPU0 fails the same fresh CUDA check while existing PointMaze
processes continue. GPU1 reports an unknown device-handle error to
`nvidia-smi`. This is a host CUDA failure, not evidence about OptiQ learning.
Preserve the stopped shard's queue/job/failure/preflight files and do not
restart or reset the server while unrelated learners run.

The existing N-Way baseline guards on 180 GPU0/GPU3 and 199 GPU3 had not
claimed any job. They were stopped; their pending 18 jobs and six OptiQ
temperature jobs are included once in the replacement plan. The new frozen
training source remains `3f54c1eb769743b430aaa79922552234603824b9`.
The new root is separate to preserve the failed preflight. The CUDA-healthy
vast-heechan-199 guards use GPU3 for 4-Way, GPU0 for 8-Way, GPU1 for
12-Way, and GPU2 for 16-Way. Each waits for the existing PointMaze and
MEOW-alpha work, actual zero compute PIDs, and the common GPU flock; no
running job is interrupted. Per-shard jobs backfill without a global barrier.
Exact jobs and supervisor commands are in `WAY_GRID_1M_FALLBACK_PLAN.json`
and `way_grid_supervisor_fallback199/`.
