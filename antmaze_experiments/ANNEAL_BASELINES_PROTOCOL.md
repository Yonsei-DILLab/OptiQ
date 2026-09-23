# AntMaze temperature annealing and missing dense baselines

User approval on2026-09-24: queue OptiQ teacher-temperature10→1,10→0.25,
10→0.5 for each v1-v4,seed0,then incomplete/missing dense baselines. Existing
jobs continue. The1M interval means environment transitions after8192 warmup,
with linear annealing and final temperature held for the rest of the native
3M/3M/4M/5M budgets. This interpretation was stated to the user before launch.

For total interactions s, T(s)=10+(Tfinal-10)clip((s-8192)/1,000,000,0,1).
All eight updates for a256-transition collection block use that block's
temperature. Warmup does not consume annealing. The first updated block has
s=8448; the first collected boundary after annealing is s=1,008,384.
The shared schedule's existing log-linear default remains unchanged; only this
profile selects the new explicit linear mode. The actual scalar passed to the
JAX update is logged at train/temperature with schedule progress and saved in
progress/result JSON. Schedule computation is stateless from environment count.

Dense negative nearest-goal distance,NovelD OFF,DACER on,256x3 actor/critic,
actorLR3e-4/criticLR5e-4,Adam,tau.005,beta1,mean-init1,random latent,N=M64,
log sigma[-5,-1]/initial-1 remain unchanged.256env,batch4096,replay1M,
8updates/256transitions,warmup8192. Native final interaction counts remain
3,008,256(v1/v2),4,008,448(v3),5,008,384(v4). Evaluate every250k with40
episodes/mode and random starting xy,save OptiQ evaluation checkpoints,final
100episodes/mode/reset and final full replay/optimizer/RNG/simulator state.

## Baseline audit and ordering

Use current dense/NovelD-OFF campaign antmaze-dense-off-16-current-s0-20260924
as the reference, never older sparse/NovelD-on runs. Inspect result/config/full
checkpoint proof and final100-episode evaluations before counting completed.
SACv2 has a completed result and final checkpoint even though its controller
recorded SIGTERM during final W&B cleanup; do not rerun it. SACv1/v3 and
DIPOv1/v3 are complete. DIPOv2/v4 are running and must not be duplicated.
MFPOv1-v4 and interrupted SACv4 are missing: queue these five fresh jobs with
native budgets/settings. SACv4 has no saved full checkpoint; preserve its old
884,736-transition partial run rather than pretending to resume it.

The latest user instruction authorizes these five baseline replacements despite
earlier holds. Preserve all old frozen sources/results/status. MFPO uses the
pinned d8b3977d29d4ef2d315e871337e5826f2eb79eb2 submodule and the validated
dependency fix. No MEOW,extra seed,sparse or NovelD-on job is registered.

Server180 owns v1/v3 (six annealing jobs,then MFPOv1/v3); server199 owns v2/v4
(six annealing jobs,then MFPOv2/v4 and SACv4). Respect GPU locks and immediately
fill each free slot in that order, without a cross-maze/method completion barrier.
Running T=1/DIPO jobs are not stopped. A job failure holds pending jobs while
preserving other live jobs. No automatic training restart. Each new job has its
own actual256env,batch4096,eight-update preflight and full-state/reward readback.

Campaign antmaze-dense-anneal-baselines-s0-20260924, W&B OptiQ/antmaze.
Server199 currently lacks outbound connectivity: use offline W&B records with
the existing managed completed-run sync sidecar. Server180 logs online.
Commit/push/share identical frozen source and prepare pinned dependencies first.
