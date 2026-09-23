# Dense AntMaze without NovelD (2026-09-23)

Latest user request: prevent the DIPO floating-point/BCE failure, switch to dense
reward without NovelD, and inspect short learned-policy trajectories before the
parallel main experiment. This supersedes the sparse profile in PROTOCOL.md.

## Reward and collection

Keep all 159 vendored DDiffPG files unchanged. Recorded changes only the returned
reward to negative Euclidean distance from next xy to the nearest original goal.
There is no sparse goal bonus, additional shaping, or reward normalization.
Original physics, reset distribution, observation, goal radius and termination
remain. NovelD is OFF in both collection and learning: zero bonus, zero RND
updates, no RND model in checkpoints. The upstream native loop calls a zero-cost
DisabledIntrinsic adapter. Its original dispatch label remains noveld solely
for compatibility, with effective_type=off and enabled=false recorded explicitly.

Each policy: seed0,256 Gym CPU environments,batch4096,replay1M,warmup8192,
8 learner updates per256 new transitions. OptiQ T.25,beta1,DACER on,mean-init1,
random latent,N=M64,log sigma[-5,-1]/initial-1,256x2GELU,LR3e-4.
SAC/DIPO/MFPO retain their resolved native architecture, objective and optimizer.
Native DIPO uses 5 diffusion steps and 20 action-gradient steps, not the old
custom adapter's 100 diffusion steps. Actor/critic LR3e-4/5e-4,clip norm1.

## DIPO safeguards and compatibility

Original float32 C51 index_add can produce probability>1 on terminal reward10/20
with support[0,5]; CPU BCE reproduces the failure. The external StableDIPO adapter
uses float64 projection accumulation, per-distribution mass normalization and
final[0,1] bounds. Preserve the upstream elementwise min of two projected targets
and BCE objective; do not renormalize that min. BCE predictions are clamped to
[float32 eps,1-eps]. Nonfinite loss/gradients are rejected before optimizer.step;
existing gradient norm1 is retained for actor,critic and target-action optimizers.
Nonfinite parameters cause an explicit failure, never silent nan_to_num masking.

Dense rewards require negative value support. DIPO uses[-6000,5],51atoms, as in
the previous dense compatibility profile; all other native settings stay fixed.
This conservative support is coarse (spacing120.1), so endpoint mass is logged.
It is a disclosed dense compatibility override, not an upstream default.

check_dense covers CPU and GPU projection/mass/BCE/finite gradients, extreme
targets and clipping, plus all4 mazes' identical physics and goal termination.
Each real job preflight uses256env,batch4096,8 learner updates; audit changed
actor/critic,zeroRND,finite metrics,full checkpoint/replay readback and actual
dense reward against next-state coordinates.

## Short trajectory gate, then main queue

Probe v1 and v3 with each of OptiQ,SAC,DIPO,MFPO:8 independent policies using
all8 GPUs,328192 total transitions=8192warmup+10000learnerupdates*32. Two DIPO
probes use faster host180. The other6 are split across remaining slots.
First inspect technical validity, learned trajectories and exploration coverage;
no success or multimodality threshold is promised at this short horizon.
Main training is registered only after the probe report is reviewed.

Main v1/v2 3008256,v3 4008448,v4 5008384 total transitions, matching the
original max_step3M/3M/4M/5M counter excluding warmup and strict>stop.
16 policies,8GPUs,independent per-job backfill every2seconds; no maze/method barrier.
Failures hold pending jobs on that host and preserve live jobs,with no automatic
restart. Old failed/cancelled queues remain inactive and their data is retained.

Evaluation every250k,interim20episodes,final100episodes per mode/reset. Full
checkpoint only at final. Direct stochastic policy,fixed identical simulator
state is primary for policy path diversity. Native controls: SAC mean,MFPO
Q-best-of10,OptiQ randomz mu-only,DIPO stochastic reverse diffusion; OptiQ zero_z
also separate. No external exploration noise or intrinsic reward during eval.
Keep failure paths, natural/fixed sets separate,training exploration separate
from final-policy rollout. W&B OptiQ/gmm-trg,explicit dense-noveld-off groups.
