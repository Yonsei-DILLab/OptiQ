# Fixed-reference geodesic reward screen

The active goal authorizes reward and hyperparameter changes while retaining
the complete OptiQ algorithm. This bounded screen tests two fresh v3 seed0
policies, T1 and T3, on otherwise free180 slots. Preserve the live v1/v4
geodesic1M experiments, all frozen sources and cancelled results.

## Evidence and reward

Original v3 goals are (-12,12) and (12,-12). The existing point-agent XY
visibility graph gives origin-to-center distances17.9864601 and16.9705627m.
Moving from (0,0) to (-.1,.1) toward the farther goal receives -14.1421 under
the old nearest-geodesic progress reward; the corresponding (.1,-.1) move
receives +14.1421. This is an initial objective asymmetry, not proof of the
cause of every historical Euclidean or geodesic collapse.

For each goal i define L_i = d_geo((0,0),goal_i) - .5 and L=min_i L_i.
The experiment-only potential is

    D(x) = min_i [ max(d_geo(x,goal_i)-.5, 0) * L/L_i ]
    reward(x,x_next) = 100 * (D(x)-D(x_next)).

The reference origin is a fixed maze constant. It does not depend on episode
history, initial randomization, chosen route or visited goals. The observation
remains the same Markov state. No success bonus, step penalty or NovelD is used.
The same two test moves now receive +12.9180 and +14.1421. Distance weights are
at most1, keeping the old conservative distance bound valid.

Two reward details change together: per-goal fixed distance normalization and
distance to the existing radius-.5 success region instead of the goal center.
The potential is zero throughout either existing success region, not forcibly
zeroed by a terminal flag. Its undiscounted telescoping total from the origin
is equal for both goals. Discount .999 still weights timing, so discounted
returns need not be equal. The old physical maze, goals, terminal success rule,
timeouts and reset are unchanged. This is point-agent XY geometry without body
inflation; it is not a collision-free full-body path or a balance guarantee.

## Controls and unchanged learning

Compare T1 with the completed v3 original-geodesic250k experiment from
438907f3a5ef681d6cde9012a66c7336fa642546. T1 differs only in this reward profile
and bookkeeping; T3 additionally changes teacher temperature from1 to3.
Neither is a resumed checkpoint or an independent seed replication.

Keep gamma .999, H/d+.7(total5.6), DACER every500learner updates, alpha .27,
alpha LR .03, noise scale .1 and GMM3/200samples. Keep actor/twin critics256x3
GELU, LR3e-4/5e-4, Adam, tau .005, random latent, N=M64, beta1, mean scale1,
log sigma[-5,-1]/initial-1. Keep256env, batch4096,8updates/256transitions,
8192warmup and replay1M. The250k post-warmup budget is258304total transitions,
250112post-warmup transitions and7816learner updates.

Seven core algorithm files must remain byte-identical to2564b59; upstream
maze/physics/reset, runner, controller, optimizer, TD, NLL and sampling remain
unchanged. Legacy reward profiles retain their values and specifications.
Record the exact source and reward specification in manifest/config/proofs.

## Execution and evidence

Commit/push/share/freeze before real preflight. Only vast-heechan-180 is used;
the inaccessible199 source copy is documented as pending. Never use4090 or
restart unknown or cancelled jobs. Existing locks and v1/v4 reservations take
precedence. Use independent slots with no all-environment completion barrier.

Each new job runs its own8448transition/8realbatch4096update preflight and
checkpoint/replay reward readback before its main learner starts. Hold pending
jobs on failure and preserve live runs. No automatic retry or extension.
W&B is OptiQ/antmaze with a distinct campaign group.

Evaluate each50k with40episodes per native/direct mode, retain intermediate
policies, and save100final episodes plus full state. v3 uses the original
fixed full starting state. Direct policy includes random latent and conditional
sigma; native mu-only is separate. No external DACER noise is added at eval.

Inspect successful routes, failures, closest-goal distance and later retention;
never treat entering two corridors as reaching two goals. Validate stored
reward sequences against this new potential, not the legacy Euclidean formula.
Any evidence is from one training seed and must be labeled accordingly.
