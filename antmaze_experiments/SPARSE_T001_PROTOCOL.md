# OptiQ sparse + NovelD, T=0.01 (2026-09-23)

Latest user instructions stop all dense running/pending jobs and launch OptiQ
v1/v2/v3/v4, one seed0 policy per maze, four jobs total first. The subsequent
correction explicitly withdraws removal of the sigma upper bound: keep log sigma
[-5,-1], initial -1, and set temperature0.01 instead. No uncapped-sigma code or
profile is introduced. Previous dense results and frozen source2633681 are kept.

Use the exact official DDiffPG sparse environment reward:0 outside a goal,
goal bonus10 or20 according to the original maze. NovelD uses the original code:
0.01 * max(n(next)-0.5*n(current),0), RND L2 novelty, normalizationfalse,
29D observations with10-band xy Fourier encoding, AdamW1e-4 and clip1.
Replay stores only environment reward; the learner recomputes the bonus.
No dense reward, additional shaping, mode-Q, HER or reward normalization.

Preserve256 environments,batch4096,replay1M,warmup8192,8learner updates per256
transitions, native budgets including original strict-stop accounting:
v1/v2 3,008,256;v3 4,008,448;v4 5,008,384 total transitions. OptiQ256x2GELU,
mean-head scale1,random normal latent,N=M64,beta1,DACERon,actor/criticLR3e-4,
gamma.99,tau.005. DACER noise_scale.1,target entropy-.9 per action dimension,
initial alpha.27,alphaLR.03,10k learner-update intervals,3GMM/200samples.

Evaluate every250k transitions:40 episodes EACH for direct policy and native
random-z mu-only. Use randomized initial xy[-2,2] for all mazes as previously
approved; preserve original training resets(v1random,v2-v4fixed). No external
DACER noise or NovelD reward during evaluation. Keep final100 episodes per
mode/reset, including fixed starts as supplementary and OptiQzero_z separately.

At every intermediate evaluation save an evaluation-only checkpoint BEFORE
rollouts:actor/critic/targets/optimizer,entropy,DACER state,and policy RNG. Record
config,step,full source SHA,file SHA256 and exact restored-Flax-state validation.
These small checkpoints omit replay and simulator state, and must never be
described as exact training-resume checkpoints. The final full replay/model/
optimizer/RND/RNG/simulator checkpoint stays enabled and verified.

Run on the four available GPUs of vast-heechan-180, one maze per GPU. Code is
committed/pushed and shared with vast-heechan-199, but no baselines or extra seeds
are registered there. Each job has its own real256env/batch4096 preflight with
8updates, sparse reward/RND/parameter/save-readback checks before fresh main
training. Preflight also checks intermediate-policy save/restore. No global
completion barrier or automatic restart; failure holds pending jobs and preserves
other live jobs. W&B OptiQ/gmm-trg, group equals campaign name.

Campaign:antmaze-sparse-noveld-t001-optiq-s0-20260923.
