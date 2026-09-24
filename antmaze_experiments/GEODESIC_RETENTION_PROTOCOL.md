# v1/v4 geodesic candidate acquisition and retention

Active-goal follow-up; no algorithm or reward implementation changes. The
completed 250k screen (source438907f3a5ef681d6cde9012a66c7336fa642546) yields:

- v1 native random starts: direct-policy successful upper17/lower9 out of100;
  mu-only upper13/lower6. This does not prove same-state multimodality.
- v4 original fixed full state: direct-policy lower73/upper25/none2 entries
  but no successes; mu-only lower61/upper39 entries and one success each.
  The two mu-only successes are weak exploratory evidence, not completion.
- v3 direct-policy right100 entries and successfulright25; do not extend that
  collapsed geodesic condition. The separate v3 Euclidean T3 run continues.

Test only v4 and v1 as two fresh seed0 runs through1M post-warmup transitions.
Only job identity/budget differ from the matched250k jobs. Do not call this an
independent replication or checkpoint resume. Compare shared checkpoints and
audit the same-seed prefix. Existing sources/results and live jobs remain intact.

Preserve gamma.999, T1, DACER H/d+.7/total5.6/interval500, geodesic reward
100*(d(current)-d(next)), no success bonus or step cost, NovelD OFF. Keep
256x3 actor/critics,3e-4/5e-4LR,Adam,tau.005,random z,N=M64,mean-init1,
log sigma[-5,-1]/initial-1,beta1,256env,batch4096,8updates/256,warmup8192,
replay1M.1008384total transitions =1000192post-warmup,31256updates and
63regulator updates. No change to physics, terminal conditions or goal locations.
Geodesic remains the existing XY point-agent potential, not full-body distance.

Every50k save evaluation-only policies and40 episodes/mode;100final and full
checkpoint. v1 native random resets;v4 original fixed full state. Direct policy
and random-z mu-only reported separately, no external DACER noise in evaluation.
Inspect500k/750k/1M success per route, failures and goal distances. Successful
v1 random-start route counts require supplementary same-state evaluation before
a conditional multimodality claim. Never mix checkpoints/modes as one policy.

Commit/push/freeze before real8448step/8batch4096-update preflight and main runs.
Verify core7files match2564b59 and reward/checkpoint readback. Use supervisor
and existingGPUlocks on180only; independent free-slot backfill preserves live
v3T3. No cancelled or unknown199 job resume, no4090 AntMaze work. Record delayed
source-only199 sharing during its SSH outage. W&B OptiQ/antmaze, separate group.
