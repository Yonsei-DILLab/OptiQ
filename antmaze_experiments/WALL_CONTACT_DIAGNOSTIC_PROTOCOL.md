# Frozen v4 wall-contact diagnostic

The completed v4 teacherfloor.5 250k policy enters upper71/lower29 routes in
100direct rollouts but reaches no goal. XY plots alone cannot establish wall
collision, falling, low speed or a remedy. Use its unchanged555bb7e checkpoint
for40native700-step CPU direct-policy rollouts at the original identical full
start. Retain randomlatent+conditional sigma and omit external DACER noise.
Do not train or change reward, physics, reset, episode limit or policy.

Run plain and instrumented conditions with identical model, full starts and RNG.
Record pre-action qpos/qvel plus actual Ant-versus-block_* contacts having
dist<=0; exclude floor contacts, self contacts, wall-wall and positive-distance
near contacts. Contact forces are not measured. Require exact equality of
paired actions, XY, returns, success and lengths; restore evaluation RNG and
verify model/optimizer/checkpoint hashes unchanged. Preserve original100-episode
primary evaluations; these40episodes are diagnostic, not a replacement.

Save all raw paths/actions/state/contact arrays, geometry, per-episode route,
success, last200step contact fraction, displacement, speed, height and remaining
geodesic distance. Interpret contact and low progress as observational evidence;
neither contact nor proximity alone establishes a causal solution or permits
changing the algorithm. No body-inflation or other reward experiment is launched.

Commit/push/share/freeze before execution. One nice19CPU supervisor service on199,
four CPU cores, no GPU use, automatic retry, new training,4090 work or cancelled
resume. Preserve live Q training on180 and all old artifacts. Load algorithm and
environment from the original555bb7e source; pin7core files to2564b59. Keep
evaluation-source provenance separate. W&B OptiQ/antmaze, evaluation-only group.
