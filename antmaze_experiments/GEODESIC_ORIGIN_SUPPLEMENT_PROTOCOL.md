# Supplementary same-state evaluation of the v1 geodesic candidate

The existing v1 geodesic gamma.999/T1/Hperdim+.7/interval500 run from training
source8e9d7d3 reaches upper19/lower16 successful routes in40direct episodes at
500224total transitions. Those primary episodes start at native random states.
They cannot alone show multiple successful paths from an identical state.

Use the existing immutable500224 policy checkpoint in
antmaze-optiq-geodesic-gamma999-retention-1m-s0-20260925 to run100episodes each
direct-policy and mu-only at the original central pose/velocity with XY=0.
Only reevaluate_fixed_origin gains an explicit --v1-origin-supplement flag;
it does not change v1 training or its primary random-start evaluation.
Construct the model/environment from frozen training source, retain original
sampler and config, and perform no training, external-noise addition or ranking.
Use exact parameter/optimizer/serialized checkpoint checks before and after,
paired identical full starting states and the frozen geodesic reward telescope.

Commit/push the evaluation source before running this inference-only job under
an autostart=false/autorestart=false low-priority CPU supervisor service on180.
No GPU allocation,4090 work or199 retry/restart. Preserve the four active
learners. Record evaluation and training source separately in provenance.
Write supplementary results in a distinct directory and W&B evaluation group;
never append them to the live native-random metric stream.

If both routes succeed, inspect a later750k/final checkpoint with the same
100episode supplementary protocol before claiming retention. All are one
training seed; repeated stochastic evaluation is not another training seed.
