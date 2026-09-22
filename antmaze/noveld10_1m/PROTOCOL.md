# NovelD10 AntMaze: 4 methods x 4 mazes x seed0 x1M

User authorization 2026-09-22: coefficient10, v1-v4, one seed, OptiQ/SAC/MEOW/
MFPO (user spelling mfpr), all8GPU queue. Exactly16fresh production runs of
1,000,000 environment interactions. Existing100k runs are not resumed.

Use branch direct-gmm-trg; commit exact code/protocol before preflight/training.
Server180 runs v1/v3, server199 runs v2/v4, four exclusive GPUslots each.
Refill any available slot every2seconds without waiting for other seeds/methods.
Existing .01baseline jobs on199 are preserved; new jobs wait for their GPUlocks.
Register all16 now. No canceled old experiments may resume. No extra seeds,
methods, environments, changed horizons or steps beyond1M. On failure preserve
live jobs, hold pending jobs, report; no auto-restart.

Campaign/service/W&Bgroup: antmaze-dense-noveld10-1m-s0-20260922.
W&B project OptiQ/gmm-trg. Each run label includes method/task/seed/c10/1m.

Compared with the existing .01 campaign source19fc37a, only the NovelD
coefficient changes to10. Keep the same DDiffPG map/low-gear MuJoCo port,
dense reward=-nearest-goal Euclidean distance, success termination<=.5m,
no success bonus or locomotion reward, timeout bootstrapping. v1randomxy reset,
v2/v3/v4fixed start; horizons500/500/700/700. No softTD, entropybackup, new
shaping, curriculum, normalization or additional exploration mechanisms.

NovelD=10*max(n(next)-.5*n(current),0), RND L2 error, normalizefalse,xyencoding
10bands, AdamW1e-4,clip1. Replay stores environmentreward only, minibatch bonus
recomputed each learner update; RND updates once per learner update.
Global coefficient default remains.01; this campaign explicitly passes10.

Oneenv,batch256,UTD1,replay1M; preserve each agent's native network/LR/optimizer.
OptiQ:256x2GELU,meaninit1,randomz,N=M64,logstd[-5,-1]/init-1,T.25,beta1,
DACERtrue,target-.9,noise_scale.1,alpha_init.27,alphaLR.03,interval10k,
GMM3/200samples,actor/criticLR3e-4,gamma.99,tau.005,warmup5k.
SAC/MEOW also warmup5k; MFPO10k. Expectedlearner/RNDupdates995k exceptMFPO990k.

Before production each local8jobs must pass272step→fresh-process280step resume,
full-state/replay/optimizer/RNG verification and actual raw evaluation audit.
Compare initial model/RND and non-ablation configs with each corresponding
.01preflight. Check coefficient linearity/default compatibility once perhost.
Reuse unchanged physics validation; do not rerun previous full physics suite.
Before completion also compare production initialconfig/hash to .01baseline.

Every25k evaluate10episodes each native and direct stochastic. NativeSAC=tanhmu,
MEOW=priorcenter,MFPO=Q-best-of10,OptiQ=randomz mu-only. Policy mode directly
samples each policy including OptiQconditional sigma; no externalDACERnoise or
NovelDreward at evaluation. Final100episodes per mode/reset; OptiQzeroz also.
Do not combine natural/fixed asindependent200starts forv2-v4.

Every100k andfinal save full replay/model/target/optimizer/entropy/DACER/RND/RNG/
simulator/counters withSHA256. Keep allsnapshots topermit futureauthorizedresume.
No automatic extension. Archive final1Mstate/replay locally, verify hashes/digests
and everyreplay environmentreward, retain rawrollouts/coverage/trainingepisodes.

Reporting prioritizes training exploration: equal-budget coveragegrowth,
0.5mvisitedbins/occupancyheatmaps,bothgoal/corridordepth,visitconcentration.
Also report native/direct-policy returns/successes,goal/routefractions and
firsttraininggoal. Keep trainingcoverage distinct from finalpolicyroutemodes.
Each method hasone trainingseed; do notclaim across-seed significance. Compare
.01baseline only at matchedsteps. Coefficient10 is a hypothesis, not aguarantee
that goalarrival remains optimal; preserve actualdense/intrinsicrewardlogs.
