# PointMaze additional OptiQ temperatures, 26 September 2026

User requests Simple/Medium/Hard OptiQ T5 andT10, one fresh seed0 each,
1M transitions. Six new runs; preserve and reuse all three completed T3 controls
from maze-deadline-1m-20260926 (source a55aaf13). No duplicate T3 training.

Use the same T3 learner, reward, network/optimizer/data profile: original DrAC
maps and sparse+100,256 parallel env,batch4096,16updates/256transitions,
warmup8192,total1,000,192transitions/62000learner calls. OptiQ256x2,N=M64,
fresh normal latent, log sigma[-5,-1],init-1,mean-init1e-4,DACER off. Only
teacher temperature differs in learning. Evaluate200k/200episodes, final500.

Latest user evaluation instruction: every OptiQ rollout uses a freshly sampled
normal z at each action, mu(s,z) only, no conditional Gaussian action noise.
Both normal and obstacle evaluations follow this; no z=0 or episode-fixed z.
Training policy sampling remains the unchanged OptiQ implementation. New
evaluation records are named mu_only/obstacle_mu_only, never mislabeled policy.
Removal SR5 uses the corrected exact subset expectation. Original T3 raw
full-policy results remain preserved; compare its saved mu_only evaluations.
Original T3 obstacle-mu data is absent and must be marked unavailable.

Use free5090 slots:46 GPUs1/3 forHard T5/T10;6 GPUs0..3 forSimple/Medium T5/T10.
Honor common GPU locks and existing learners; no preemption or automatic retry.
Commit/push/share frozen source and this plan before each original8448step/
16update preflight and fresh main run. Independently backfill without barriers.
Record source SHA, commands, raw rollout arrays, final checkpoints/replay,
automatic bold-line figures and SHA proofs. Failures pause pending work.

All new/report trajectory lines use1.8pt and alpha.7, preserving the exact
rollout subset and counts. Default figures and metrics use OptiQ mu-only;
historical sigma-included material is supplementary and never relabeled.
