# Conditional sigma upper-bound grid with fixed initialization

Latest explicit user approval (2026-09-25): test log sigma upper bounds
0,1,2,3,and no upper bound. Apply to the ongoing v3/v4 problem: ten fresh seed0
runs,250k postwarmup each. Reuse archived cap-1 O/P controls. Hold initial log
sigma-1 and lower-5 in every condition. The earlier cap-2/-3 screen changed both
cap and initialization; it is not an isolated upper-bound comparison.
This is an AntMaze ablation; shared defaults remain[-5,-1]/initial-1.

The finite raw scale maxima are1,2.718282,7.389056,20.085537. Actions remain
box-truncated within[-1,1]; increasing raw sigma does not enlarge action bounds.
The unbounded profile passes positive infinity to the existing upper-clip
parameter, genuinely removing only its upper limit. No substituted finite cap,
sampler rewrite or objective change is allowed. Saved config records explicit
actor_sigma_upper_bound_removed=true and actor_sigma_upper_bound=null, while
the native runtime config retains+infinity. Python JSON preserves that runtime
value as Infinity; consumers should use the explicit flag/null for portability.

Compare with archived O/P: v3 normalized remaining geodesic progress100/T3/
teacherfloor1 and v4 original geodesic progress100/T1/floor.5. Keep gamma.999,
H/d+.7/interval500,DACER on,random normal latent,N=M64,256x3GELU,meaninit1,
Adamactor3e-4/critic5e-4,tau.005,batch4096,replay1M,256env/eightupdates,warmup8192,
NovelD OFF,no bonus/no step cost,native700/fixed full origin. Resulting258304
transitions/7816updates/16regulator updates match controls. Forty direct+forty
native episodes each50k,100final per mode and full checkpoint/replay. Primary
trajectory evidence is direct conditional-noise policy, with mu-only supplementary.
Judge both successful routes and their retention, not broad entrances alone.

Schedule v3 five conditions on180,v4 five on199,using eight5090 GPUs and existing
locks. Each slot backfills independently every2seconds; no completion barrier
between methods, mazes or preflights. Preserve all live/frozen/archived results.
No4090/cancelled resume/extra seed/automatic retry or extension. Failure holds
that host's pending jobs and preserves its live jobs for inspection.

Higher conditional scale can increase jitter, or represent dispersion through
sigma instead of latent modes. Cap saturation alone does not establish benefit.
The v3 control's direct44/100 versus mu-only80/100 already warns about accuracy.
DACER may respond to changed policy entropy; its mechanism and settings stay fixed.
The frozen teacher MC diagnostic's eight states/eight clouds gave M64 ESS32.6
(v3)/26.2(v4), not one-candidate collapse. Its output-gradient variability is not
actual batch4096 parameter-gradient variance; N/M remain64.

Preflight: require exact initial actor/critic equality to O/P, actual8updates
withbatch4096, full replay reward/readback and model/optimizer/RNG preservation.
Scratch independent sigma heads at raw log scales-6,-1,4,8 verify actual clipping
or no upper bound, bounded finite draws, unchanged marginal NLL and finite output
gradients, and density normalization versus float64. These are finite-range
checks, not a guarantee for arbitrary unbounded float32 values. The existing
runner rejects nonfinite actions and learner metrics. At sufficiently large
raw log sigma, CDF cancellation/overflow may fail; preserve and report such a
failure rather than silently cap or alter the algorithm.

All nine pinned computational files and upstream remain unchanged. Add explicit
configuration profiles and validation only. Commit/push/share/freeze code and
protocol before preflight or training. Log to W&B OptiQ/antmaze; retain manifests,
runtime proofs and failed as well as successful raw rollouts per condition.
