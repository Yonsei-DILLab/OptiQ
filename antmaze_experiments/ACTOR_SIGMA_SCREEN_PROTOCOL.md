# Conditional sigma cap and initialization screen

The active reward/hyperparameter goal authorizes this bounded ablation; no
algorithm structure is changed. The complete entropy grid did not retain both
successful routes. Later teacher-floor controls improve v3 right-goal learning
but lose the left; v4 retains entrances without goal acquisition. Fixed64 did
not help. env32 at250112 has right16/40 v3 successes and lower40/40 v4 entries
with no v4 success. These findings do not establish a causal sigma failure.

Compare four fresh seed0 policies on unlocked199GPU slots: v3/v4 x actor
log_std cap and initial value {-2,-3}. Keep the lower bound-5. These correspond
to initial/maximum conditional scales .135335 and .049787 versus control
.367879. The cap and initial value change together and cannot be attributed
individually. Default[-5,-1]/initial-1 stays unchanged for every other profile.

Use the original256env/eight-update controls, not the env32 treatment. v3 uses
start-normalized remaining geodesic progress100,T3,teacherfloor1; v4 original
geodesic progress100,T1,teacherfloor.5. Both gamma.999,H/d+.7/500,DACER on,
random normal latent,N=M64,256x3GELU,meaninit1,Adam actor3e-4/critic5e-4,tau.005,
batch4096,replay1M,warmup8192,NovelD off,no bonus or step cost. The teacher's
existing exploration floor, sampling/density correction, loss/TD/optimizer,
action bounds, physics, resets, observations and native700 horizon stay fixed.
DACER may react to changed entropy; its settings/mechanism are unchanged.

Hypothesis: reducing state-conditional Gaussian spread may improve precise
movement near the goals while retaining latent-dependent behavior. Lower noise
could instead hinder discovery or increase collapse. Direct-policy evaluation
always samples the actual trained conditional sigma; no post-hoc noise scaling
or mu-only substitution is permitted. Mu-only remains a separate supplement.

Budget250k postwarmup =258304 total/7816 learner updates/16 regulator updates.
Evaluate40direct+40native at50176/100096/150016/200192/250112; final100 per mode,
full checkpoint/replay. v3/v4 use the original identical full start. Preserve
all failures. Both successful routes, then later retention, are needed for the
goal; a few entrances alone are not a solution. No automatic extension/retry.

The adapter forwards existing log_std_min/max/initial_log_std after validating
the shared MuJoCo default. All nine computational files stay byte-identical to
2564b59; upstream antmaze/, reward, physics and shared defaults are unchanged.
The only new validation generalizes an existing sigma-bound assumption when
an explicit ablation is present. No algorithm flag or optimization is added.

Commit/push/share/freeze before registration or real preflight. Each job must
pass8448transitions/eightupdates/batch4096 GPU preflight, checkpoint/full replay
reward readback, direct/native sampler and serialization tests. Its initial
critic must equal the archived O/P control exactly. Replacing only the initial
sigma-head bias with-1 in an isolated scratch tree must reproduce the archived
control actor hash. This proves all other initial actor parameters agree.
Scratch checks must preserve parameters, optimizer, counters and RNG.

Use supervisor/autostartfalse/autorestartfalse and existing GPU locks/backfill.
Preserve env32 final evaluations on180 and every frozen/stopped campaign. No
cancelled resume,4090 work,extra seed or automatic extension. W&B OptiQ/antmaze,
group antmaze-optiq-sigmacap-v34-250k-s0-20260925. Controller holds pending on
failure and preserves other live jobs. Save source/config/raw paths, full-state
proofs, initial/final actor sigma verification and post-hoc reporting provenance.
