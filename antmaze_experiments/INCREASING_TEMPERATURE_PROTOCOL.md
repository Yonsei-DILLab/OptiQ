# Acquisition followed by a softer teacher temperature

The active goal permits reward/hyperparameter changes while preserving the
OptiQ algorithm. The fixed-T1 original-geodesic v1 policy succeeds on both
routes at500k from the same origin (49upper/40lower out of100), but at1M all100
use the lower successful route. The corresponding v4 candidate loses the
upper route before attaining useful success; at550k all40 direct episodes
take the lower route, with26 successes. That v4 run was screened and stopped.
Conversely, fixed T3 at250k in v4 preserves38upper/45lower entries but has no
successes and a final distance around10.6m. Fixed T10 is even slower.

Test fresh seed0 v4 and v1 with the EXISTING linear teacher-temperature formula,
T1 at the end of8192warmup, rising to T3 over the next1M transitions. Hold3
thereafter. Hypothesis: earlier stronger Q preference helps acquire useful
behavior, while later softer preference may retain alternatives. This can
also hurt acquisition or fail to preserve either route; it is not a solution
until successful multi-route retention is observed.

The interpolation function, actor/critic losses, TD backup, optimizer,
sampling, latent prior, DACER regulator and seven pinned core files are
unchanged. Only the existing schedule parser/CLI positivity validation is
generalized to permit increasing endpoints. The controller verifies the
explicit temperature schedule alongside the explicit DACER target instead of
requiring a constant temperature. No new temperature rule, optimization step
or algorithm term is introduced. Exact comparisons against the frozen parent
verify that all previously valid decreasing schedules produce identical
values. The interpolation function source itself must remain unchanged.

Compare with original-geodesic gamma.999/T1 source
8e9d7d3c2c79f797654ccfb21913d2c000b89f71 at matched checkpoints. Only job
identity and the temperature schedule differ. Do not use the v3 normalized
reward in this campaign. Keep100*(nearest-geodesic-distance decrease), no
bonus/time cost/NovelD, gamma.999, DACER H/d+.7/500updates,alpha.27,
alphaLR.03,noise scale.1,GMM3/200;256x3 actor/twin critics,LR3e-4/5e-4,
Adam,tau.005,random latent,N=M64,beta1,mean-init1,log sigma[-5,-1]/initial-1,
256env,batch4096,8updates/256,warmup8192,replay1M. Physics, native reset and
termination are unchanged. No old snapshot or result is overwritten.

Maximum1M post-warmup (1008384total/31256updates/63regulator updates). Save
40episodes per mode and evaluation policies each50k,100final plus full state.
v4 uses original fixed full state; v1 primary remains native random starts.
Direct evaluation retains random latent and conditional sigma, with no
external DACER noise. Mu-only is supplementary. Evaluate promising v1 saved
checkpoints additionally at the original full origin before claiming
conditional multi-route retention; random-start results alone are insufficient.

Inspect250k/500k/750k and stop sustained minority-route loss rather than
automatically using the whole budget. Preserve every failure and stopped
artifact; no automatic retry or budget extension. One training seed is an
exploratory candidate, not broad replication evidence.

Commit/push/freeze before actual per-job8448transition/8batch4096update
preflights. Require source/config, expected live temperature1.000512 at
the preflight endpoint, initial-parameter identity, regulator-count and full
replay/checkpoint verification before main learning. Use independent180
slots and existing locks, preserving the live v3 discount pair and v4T10.
199 source sharing is pending SSH recovery; no unknown-job restart or4090
AntMaze use. W&B OptiQ/antmaze, separate campaign group.
