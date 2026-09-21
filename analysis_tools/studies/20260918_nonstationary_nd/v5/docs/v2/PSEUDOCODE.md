# OptiQ v2 — final implementation specification

확정일: 2026-09-11. 기준: **continuous-latent checked K64**, training commit
`8cb4f237aacb61113f65e693e23236f17d77f978`, Humanoid-v4 seeds 0,1,2,3.
900K–1M 평균 5362.67 / 최종 정책 새 평가 5397.09를 기록한 구성이다.
이 문서는 framework-independent 구현 명세다. `Normal`, `logsumexp`, `grad`,
`stop_gradient`, `Adam`은 사용하는 framework의 대응 연산으로 구현한다.
수치 함수의 독립 NumPy 예시는 [reference_math.py](reference_math.py)에 있다.

## 0. 고정된 알고리즘의 범위

실행 정책은 continuous Gaussian latent를 사용하는 semi-implicit tanh-Gaussian이다.
Critic은 scalar twin-Q다. Teacher는 actor의 조건부 Gaussian을 혼합한 proposal이다.
Full beta=1 importance weights와 16×64 entropic OT로 조건부 Gaussian NLL을 만든다.
IDAC 엔트로피 추정을 soft TD backup에 사용하고, 별도 표본 soft-score 검사로
actor 업데이트를 수락/거절한다. 유한 latent codebook, proximal η,
온도 annealing, ESS에 따른 온도/β 조절, 추가 uniform 탐색은 사용하지 않는다.

원안과 달라진 것은 teacher(실현 샘플 KDE → learned conditional mixture),
projection(argmax MSE → full conditional NLL), sampled soft guard다.
Mean-Q teacher는 원래 설정에서 유지한 선택이다. 이것은 IDAC 전체 알고리즘의 재현도,
정확한 W2를 직접 최소화하는 actor도 아니다.

## 1. 상수, 공간, tensor shape

```python
TRAIN_SEEDS = [0, 1, 2, 3]  # independent training runs, not vector environments
ENV = 'Humanoid-v4'
TOTAL_ENV_STEPS = 1_000_000
WARMUP = ACTOR_START = 5_000
N_ENVS = 1
B = 256                   # training replay batch
N = 16                    # student / sampled conditional components
K = 64                    # teacher candidates; R=4 only determines K=N*R
M = 16                    # entropy mixture: generating component + 15 auxiliaries
BG = 32                   # separate replay batch for guard
J = 8                     # independent action draws per guard state
GAMMA = 0.99
TARGET_TAU = 0.005         # target <- (1-tau)*target + tau*live
T = 0.1                   # both Boltzmann temperature and TD entropy coefficient
BETA = 1.0
TEACHER_SIGMA_FLOOR = 0.05 # pre-tanh, teacher only
OT_EPSILON = 0.25          # not the Boltzmann temperature
OT_ITERATIONS = 100
LR_ACTOR = LR_CRITIC = 3e-4
ADAM_B1 = 0.9; ADAM_B2 = 0.999
ADAM_EPS = 1e-8; ADAM_EPS_ROOT = 0
MAX_GLOBAL_GRAD_NORM = 2.0
REPLAY_CAPACITY = 1_000_000
UTD = POLICY_DELAY = 1
LOG_STD_MIN = -5.0; LOG_STD_MAX = 1.0
INITIAL_LOG_STD = log(0.5)
GUARD_SE_MULTIPLIER = 0.0
DTYPE = float32
```

`O` denotes observation dimension; `D` is action dimension and latent dimension.
Humanoid-v4: O=376, D=17, environment action bounds approximately [-0.4,0.4].
Other environments must read dimensions/bounds from their spaces.
Use Gymnasium 0.29.1 / MuJoCo 2.3.7 and the default Humanoid-v4 constructor:
forward_reward_weight=1.25, ctrl_cost_weight=0.1, healthy_reward=5,
terminate_when_unhealthy=true, healthy_z_range=(1,2), reset_noise_scale=.01,
exclude_current_positions_from_observation=true. Do not substitute a v3/v5 XML,
observation definition or reward convention when comparing the reported score.

| Object | Shape | Meaning |
|---|---|---|
| replay state / action | [B,O] / [B,D] | action in normalized coordinates |
| student z, μ, logσ, u, a | [B,N,D] | independent latent/noise per state and component |
| teacher v, b | [B,K,D] | pre-tanh and normalized action respectively |
| teacher log q, Q, weights | [B,K] | normalization over K separately per state |
| cost C, plan P, row probabilities R | [B,N,K] | no cross-state transport |
| twin critic output | [2,B,1] | last dimension is scalar, not categorical mass |
| guard gaps | [J,BG] | keep the state grouping when estimating standard error |

The library's raw critic tensor has a trailing singleton dimension. The
`CRITIC` helper below removes it and returns [batch], so rewards, terminal masks,
log densities and TD targets all have shape [batch]; never broadcast [batch,1]
against [batch]. The pseudocode uses these shape helpers:

```python
def broadcast_states(s, count):
    return broadcast(s[:,None,:], [len(s),count,s.shape[-1]])
def repeat_states(s, count):
    return broadcast_states(s,count).reshape(len(s)*count,s.shape[-1])
def flatten(x):
    return x.reshape(-1,x.shape[-1])
def gather_per_state(x, indices):
    return x[arange(len(x))[:,None], indices]  # [B,N,D], [B,K] -> [B,K,D]
def min_twin_Q(critics, s, a):
    return minimum(CRITIC(critics.params[0],s,a), CRITIC(critics.params[1],s,a))
```

All policy densities, entropies, OT costs and replay actions use normalized
action coordinates (-1,1)^D. Environment actions are
`low + (a+1)*(high-low)/2`; normalize warmup actions by the inverse transform.
Do not multiply only the TD entropy density by the environment action scale.
That changes its coordinate convention and the soft backup. There is no
observation normalization, reward normalization/scaling, action repeat override,
BN, LN or dropout in the final configuration.

## 2. Networks and optimizer initialization

```python
def ACTOR(theta, s, z):
    h = concat(s, z, axis=-1)
    for layer in theta.hidden:       # widths [256,256,256]
        h = gelu_tanh_approx(linear(h, layer.W, layer.b))
    mu = linear(h, theta.mu.W, theta.mu.b)
    log_std = clip(linear(h, theta.log_std.W, theta.log_std.b), -5, 1)
    return mu, log_std

def CRITIC(phi_k, s, a):
    h = concat(s, a, axis=-1)
    for layer in phi_k.hidden:       # widths [256,256,256]
        h = gelu_tanh_approx(linear(h, layer.W, layer.b))
    return linear(h, phi_k.out.W, phi_k.out.b)[...,0]  # unbounded scalar [batch]
```

GELU follows `flax.linen.gelu`/JAX's default approximation:
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))`.
Actor hidden kernels use variance-scaling uniform, fan-average, scale=1:
`W ~ Uniform(-sqrt(6*scale/(fan_in+fan_out)), +sqrt(...))`.
The μ output uses scale=1e-4; this is a **variance scale**, not an SD of 1e-4.
All those biases are zero. Logσ output kernel is exactly zero and its bias is
`log(0.5)`, so the initial policy σ is 0.5 regardless of observation size.
There is no fixed z-to-μ residual connection.

The two critics have separate parameters and independent initialization.
Every critic Dense uses Flax's default LeCun-normal initializer: variance 1/fan_in,
implemented via a Normal truncated at ±2 with correction 0.87962566103423978.
Specifically `W=truncated_standard_normal(-2,2)/[sqrt(fan_in)*0.87962566103423978]`.
Critic biases are zero. Copy live critics into target critics at initialization.
Do not use a framework's default Linear initialization without matching these choices.

Actor and both critics have separate Adam optimizer states. The critic optimizer
owns both networks together. Before Adam, clip the gradient tree by one global L2
norm: if gnorm>2, multiply every leaf by 2/gnorm. For critics this norm includes
**both networks**. Use Adam betas (.9,.999), bias correction, epsilon outside sqrt
1e-8, no weight decay and constant learning rate. Rejected actor proposals restore
parameters, Adam first/second moments and Adam step together.

```python
def CLIP_GLOBAL(gradient_tree, limit):
    norm = sqrt(sum(sum(leaf*leaf) for leaf in leaves(gradient_tree)))
    return gradient_tree if norm < limit else tree_scale(gradient_tree,limit/norm)

def ADAM_STEP(state, gradients, lr):
    # Operations are applied leafwise; return a new state (do not mutate old).
    step = state.step + 1
    m = .9*state.m + .1*gradients
    v = .999*state.v + .001*gradients**2
    m_hat = m/(1-.9**step); v_hat = v/(1-.999**step)
    params = state.params - lr*m_hat/(sqrt(v_hat)+1e-8)
    return TrainState(params=params,m=m,v=v,step=step)
```

## 3. Policy sampling and joint mixture density

```python
def SAMPLE_POLICY(theta, s, rng):
    z = rng.normal(shape=[len(s), D])
    eps = rng.normal(shape=[len(s), D])       # independent of z
    mu, log_std = ACTOR(theta, s, z)
    u = mu + exp(log_std)*eps
    return tanh(u), u

def COMPONENTS(theta, s, count, rng):
    z = rng.normal(shape=[len(s), count, D])
    mu, ls = ACTOR(theta, broadcast(s[:,None,:], [len(s),count,O]), z)
    return z, mu, ls

def LOG_TANH_JACOBIAN(u):
    return sum(2*(log(2)-u-softplus(-2*u)), axis=-1)

def LOG_MIXTURE(u, mu, ls):
    # u [B,K,D], mu/ls [B,M,D]; output [B,K].
    delta = (u[:,:,None,:] - mu[:,None,:,:])*exp(-ls[:,None,:,:])
    component_log_prob = sum(
        -0.5*delta**2 - ls[:,None,:,:] - 0.5*log(2*pi), axis=-1)
    return (logsumexp(component_log_prob, axis=-1) - log(mu.shape[1])
            - LOG_TANH_JACOBIAN(u))
```

`LOG_MIXTURE` sums action-coordinate log densities **before** logsumexp over
components. Mixing each coordinate separately gives a different policy.
Keep the original pre-tanh u/v even when float32 tanh saturates to ±1.
Never reconstruct u with atanh(tanh(u)), and never replace the stable Jacobian
by a clipped-action approximation in density correction.

The actual policy is `pi(a|s)=E_z[k_theta(a|s,z)]` under continuous z~N(0,I_D).
N=16 is a training sample count. It is not a 16-element latent prior.
Inference needs one actor forward plus Gaussian noise and tanh; no Q, OT or
mixture-density computation is required to execute an action.

## 4. IDAC action and density for soft TD

```python
def IDAC_ACTION_AND_LOG_DENSITY(theta, s, rng):
    z, mu, ls = COMPONENTS(theta, s, count=M, rng=rng)
    eps = rng.normal(shape=[len(s),D])
    u = mu[:,0,:] + exp(ls[:,0,:])*eps
    a = tanh(u)
    log_g = LOG_MIXTURE(u[:,None,:], mu, ls)[:,0]
    return a, log_g
```

Component 0 generates the action and remains in the M=16-component density.
The other 15 latents are independent of it. Use the **actual** actor logσ,
without teacher σ floor. The sampled action's marginal is the executed policy;
the mixture density estimate is not its exact continuous-latent marginal density.

For fixed s and theta, `E[-log_g] <= H(pi_theta)` over latent/action draws.
The bound is in expectation, not per sample. Fixed M generally leaves bias;
more TD minibatches at fixed M do not automatically remove that bias.

## 5. Critic update — current policy, target min-Q

```python
def UPDATE_CRITIC(actor, critics, target_critics, batch, rng):
    s, a, r, s_next, terminal = batch
    with no_grad:
        a_next, log_g_next = IDAC_ACTION_AND_LOG_DENSITY(actor.params, s_next, rng)
        q_next = minimum(CRITIC(target_critics[0], s_next, a_next),
                         CRITIC(target_critics[1], s_next, a_next))
        y = r + GAMMA*(1-terminal)*(q_next - T*log_g_next)

    def loss(phi):
        q1 = CRITIC(phi[0], s, a)
        q2 = CRITIC(phi[1], s, a)
        return mean_over_B((q1-y)**2) + mean_over_B((q2-y)**2)

    # SUM of twin losses, not mean over 2*B and not half-MSE.
    g = grad(loss)(critics.params)
    critics = ADAM_STEP(critics, CLIP_GLOBAL(g, 2), lr=3e-4)
    target_critics = .995*target_critics + .005*critics.params
    return critics, target_critics
```

Terminal means true MDP termination. A time-limit-only truncation still
bootstraps from the final observation, not the reset observation. Store both
termination and truncation flags, or correctly reproduce SB3's timeout mask.
At a true terminal y=r. Humanoid-v4's standard horizon is 1000 steps.
TD action smoothing is disabled; no teacher noise is added to a_next.
The scalar target is not clipped to [-3600,3600]; those retained categorical
configuration fields have no scalar support/clipping effect.

## 6. Teacher proposal and importance weights

Use the **updated live critics** from section 5 and the pre-update actor.
The actor states here are the same B states as the critic update, not BG guard states.

```python
def MAKE_STOPPED_TEACHER(actor_params, critics, s, rng):
    z, mu_old, ls_old = COMPONENTS(actor_params, s, N, rng)
    eps_student = rng.normal(shape=[B,N,D])
    u_student = mu_old + exp(ls_old)*eps_student
    a_student = tanh(u_student)

    with no_grad:
        proposal_mu = stop_gradient(mu_old)
        proposal_ls = maximum(stop_gradient(ls_old), log(.05))
        indices = rng.integers(low=0, high=N, shape=[B,K])
        eta = rng.normal(shape=[B,K,D])
        # Every candidate independently selects a uniformly weighted component.
        v = gather_per_state(proposal_mu, indices) + exp(
            gather_per_state(proposal_ls, indices))*eta
        b = tanh(v)
        log_q = LOG_MIXTURE(v, proposal_mu, proposal_ls)

        q1 = CRITIC(critics.params[0], repeat_states(s,K), flatten(b)).reshape(B,K)
        q2 = CRITIC(critics.params[1], repeat_states(s,K), flatten(b)).reshape(B,K)
        q_teacher = (q1+q2)/2
        logits = q_teacher/T - log_q         # beta exactly 1
        w = softmax(logits, axis=-1)         # stabilize by per-row max subtraction
        ess = 1/sum(w*w, axis=-1)            # diagnostic, not a controller
        C = sum((a_student[:,:,None,:]-b[:,None,:,:])**2, axis=-1)
        P = SINKHORN(C, w)
        R = P/maximum(sum(P,axis=-1,keepdims=True), 1e-20)
        diagnostics = {'ess_per_state':ess, 'teacher_log_q':log_q,
                       'teacher_q':q_teacher, 'twin_disagreement':abs(q1-q2),
                       'ot_row_error':mean(abs(sum(P,axis=-1)-1/N)),
                       'ot_column_error':mean(abs(sum(P,axis=-2)-w))}

    return z, stop_gradient(v), stop_gradient(R), diagnostics
```

The proposal is explicitly defined **before sampling** by saved μ/logσ.
Sampling and density use the same floored scales, and density sums all N
components, not just the selected component. This is not a KDE centered on
realized student u. No student anchor, hard perturbation cutoff, rejection,
top-k selection, gradient ascent on candidates or Gaussian diffusion steps are used.
R=4 sets the total K=64; it does not allocate exactly four draws to each component.
Finite self-normalized importance sampling approximates Boltzmann extraction;
the 64 candidates are not IID draws from the Boltzmann target.

For fixed proposal parameters F, beta=1 cancels q_F in the ideal weighted density:
`q_F(a)*exp(Q_mean(a)/T)/q_F(a) = exp(Q_mean(a)/T)`.
All Gaussian kernels have full support before tanh. Full support does not imply
that a finite candidate cloud covers all useful action regions.

## 7. Exact Sinkhorn convention used by this implementation

```python
def SINKHORN(C, w):              # C [B,N,K], w [B,K]
    columns = clip(w, 1e-20, 1)
    columns = columns/sum(columns,axis=-1,keepdims=True)
    log_rows = full([B,N], -log(N))
    log_columns = log(columns)
    log_kernel = -C/0.25
    log_u = zeros([B,N]); log_v = zeros([B,K])
    for iteration in range(100):
        log_u = log_rows - logsumexp(log_kernel+log_v[:,None,:], axis=-1)
        log_v = log_columns - logsumexp(log_kernel+log_u[:,:,None], axis=-2)
    return exp(log_kernel+log_u[:,:,None]+log_v[:,None,:])
```

The cost is raw **sum** of squared normalized-action differences, not its
coordinate mean, not half the squared norm, and not normalized by batch/state
mean cost. Entropic OT corresponds to `<P,C> + epsilon*sum(P*(log(P)-1))` with
uniform row masses and corrected column masses. Multiplying C without scaling
epsilon changes the algorithm. Finite 100 iterations need not enforce row
marginals exactly; the last update enforces columns more closely. Normalize rows
to R for NLL even if marginal residuals are nonzero. Log both row and column errors.
Do not replace the final plan with hard argmax assignments.

## 8. Full conditional OT NLL and the candidate actor

```python
def PROPOSE_ACTOR_UPDATE(actor, critics, s, rng):
    z, v, R, metrics = MAKE_STOPPED_TEACHER(actor.params, critics, s, rng)
    # z, v, R are fixed for this optimizer step. No redraw inside loss/grad.
    teacher_mean = einsum('bnk,bkd->bnd', R, v)
    teacher_var = maximum(einsum('bnk,bkd->bnd',R,v*v)-teacher_mean**2, 0)

    def loss(theta):
        mu, ls = ACTOR(theta, broadcast_states(s,N), z)
        nll_per_coordinate = (0.5*((mu-teacher_mean)**2 + teacher_var)*exp(-2*ls)
                              + ls + 0.5*log(2*pi))
        return mean_over_B_N(sum(nll_per_coordinate, axis=-1))

    g = grad(loss)(actor.params)
    candidate = ADAM_STEP(actor, CLIP_GLOBAL(g, 2), lr=3e-4)
    return candidate, metrics
```

This is exactly the row-weighted conditional Gaussian NLL:
`-(1/(B*N))*sum_bij R_bij*log Normal(v_bj;mu_bi,diag(exp(2*ls_bi)))`.
The moment form avoids allocating [B,N,K,D]. The logσ term has a **positive** sign;
do not drop teacher variance or average over D. Teacher tanh Jacobian is constant
with respect to the new theta for fixed v, so it does not enter this loss.
Both μ and σ receive supervised gradients. All Q evaluation, proposal parameters,
teacher samples, importance weights, OT costs/plan and target moments are stopped.
No gradient propagates through Q, Sinkhorn, teacher sampling or teacher density.
There is no additional actor entropy loss or reverse-KL loss.

Full conditional NLL retains each OT row's mean and variance; a single Gaussian
conditional is still a projection and cannot in general represent a multimodal
row exactly. NLL decrease is not proof of exact W2 decrease or policy improvement.

## 9. Sampled soft-score guard and rollback

The validation states are a separate uniform replay draw of size BG=32 made
before the critic/actor update. The two replay batches may overlap by chance;
they are not forced disjoint and are not fresh environment rollouts.

```python
def ENTROPY_BRACKET(theta, states, z_17, eps):
    # z_17 [BG,M+1,D]: M+1=17 latents, NOT 16 total.
    mu, ls = ACTOR(theta, broadcast_states(states,M+1), z_17)
    u = mu[:,0,:] + exp(ls[:,0,:])*eps
    h_lower = -LOG_MIXTURE(u[:,None,:],mu[:,:M,:],ls[:,:M,:])[:,0]
    h_upper = -LOG_MIXTURE(u[:,None,:],mu[:,1:,:],ls[:,1:,:])[:,0]
    return tanh(u), h_lower, h_upper

def ACCEPT_OR_ROLLBACK(old, candidate, frozen_live_critics, states, rng):
    gaps = zeros([J,BG])
    for j in range(J):
        z = rng.normal(shape=[BG,M+1,D])
        eps = rng.normal(shape=[BG,D])
        # EXACT SAME base z/eps on both sides (common random numbers).
        a_old, _, h_upper_old = ENTROPY_BRACKET(old.params,states,z,eps)
        a_new, h_lower_new, _ = ENTROPY_BRACKET(candidate.params,states,z,eps)
        q_old = min_twin_Q(frozen_live_critics,states,a_old)
        q_new = min_twin_Q(frozen_live_critics,states,a_new)
        gaps[j] = (q_new+T*h_lower_new)-(q_old+T*h_upper_old)

    state_gaps = mean(gaps,axis=0)
    mean_gap = mean(state_gaps)
    se = std(state_gaps,ddof=1)/sqrt(BG)
    margin = mean_gap - 0.0*se
    accept = all(isfinite(gaps)) and isfinite(margin) and margin > 0
    selected = candidate if accept else old
    return selected, {'accept':accept,'gap':mean_gap,'se':se,
                      'negative_state_fraction':mean(state_gaps<0)}
```

Lower density uses z[0:16], including generating z0. Upper density uses z[1:17],
excluding it: these 16 latents are independent of the action. The two estimates
overlap in auxiliaries z1..z15. The bounds hold after expectation; their sampled
ordering can reverse. The old/new actions generally differ even with the same
base random draws. Use the same **updated live minimum twin-Q** for both sides;
do not compare scores under different critic versions or use teacher mean-Q here.

The completed run has **no extra SE confidence margin** (multiplier=0), no per-state
all-positive requirement and no multiple-step line search. On rejection restore
the whole old actor/Adam TrainState; keep the critic update. Consumed RNG and the
number of attempted updates do not roll back. Target actor parameters are copied
from the accepted actor afterward (policy_tau=1); this target actor is not used
by the semi-implicit TD backup.

## 10. Full environment/training loop and evaluation order

```python
initialize actor, twin critics, critic targets, separate Adam states
initialize replay capacity 1_000_000, environment seeded with TRAIN_SEED
initialize learner RNG, policy-action RNG and NumPy replay RNG from TRAIN_SEED
s = env.reset(seed=TRAIN_SEED)
for t in range(1, 1_000_000+1):
    if t <= 5_000:
        env_action = env.action_space.sample()
        a = NORMALIZE_ACTION(env_action)
    else:
        a_batch, _ = SAMPLE_POLICY(actor.params, s[None,:], policy_rng)
        a = a_batch[0]                       # one normalized [D] action
        env_action = UNSCALE_ACTION(a)
    s_final, reward, terminated, truncated, info = env.step(env_action)

    # Historical callback: after env.step, before this step's replay/train.
    if t == 1 or t % 5_000 == 0:
        EVALUATE_CURRENT_POLICY(10 stochastic episodes)

    replay.add(s,a,reward,s_final,terminal=terminated,timeout=truncated and not terminated)
    s = env.reset() if terminated or truncated else s_final
    # VecEnv implementations must recover terminal_observation before add().
    if t > 5_000:
        batch = replay.sample_uniform_with_replacement(256)
        guard_states = replay.sample_uniform_with_replacement(32).states
        critics, targets = UPDATE_CRITIC(actor,critics,targets,batch,learner_rng)
        old = actor                         # immutable snapshot including Adam
        candidate, metrics = PROPOSE_ACTOR_UPDATE(old,critics,batch.states,learner_rng)
        actor, guard_metrics = ACCEPT_OR_ROLLBACK(old,candidate,critics,
                                                  guard_states,learner_rng)
        target_actor_params = actor.params  # policy_tau=1; not used in TD
        attempted_updates += 1
        if t % 50_000 == 0:
            SAVE_ACTOR_AND_CRITIC_TRAINSTATES(t)
SAVE_ACTOR_AND_CRITIC_TRAINSTATES(1_000_000)
```

The first update occurs after environment step 5001. At 1M there are 995000
critic updates/actor proposals, while actor Adam's step counts only accepted
updates. This algorithm uses one environment step followed by one critic update
and one actor proposal, not UTD=2 or one update per episode.
Training evaluation at step t observes the actor **before** update t; a checkpoint
named t stores the actor **after** update t. Do not equate those policies.

### Randomness details for matching this repository

All z's and Gaussian noises are independently drawn across states/components and
uses, except the deliberately shared old/new guard randomness. TD action draws,
actor student/teacher draws and guard draws use different keys. Teacher μ/logσ
reuse the student's latent set, not the TD or guard set.

The literal JAX split tree is:

```python
# Seed setup (policy.build, then DIME setup):
key = jax.random.key(seed)
key, actor_init_key, critic_init_key, dropout_key, bn_key = split(key,5)
key, policy_key = split(key,2)
policy_key, noise_key = split(policy_key,2)       # initial reset_noise()
key, entropy_dummy_init_key = split(key,2)       # retained const-coef initialization

# Every policy predict(), including evaluation:
policy_key, noise_key = split(policy_key,2)
z_key, eps_key = split(noise_key,2)

# Every critic update:
key, actor_key, unused_noise, target_dropout, current_dropout, redq_key = split(key,6)
td_z_key, td_eps_key = split(actor_key,2)

# Every actor proposal:
key, latent_key, proposal_key, dropout_key = split(key,4)
student_z_key, student_eps_key = split(latent_key,2)
teacher_component_key, teacher_eps_key = split(proposal_key,2)

# Every guard, after the proposal:
key, guard_key = split(key,2)
draw_keys = split(guard_key,8)
for draw_key in draw_keys:
    guard_z_key, guard_eps_key = split(draw_key,2)
    # reuse these exact keys for old and candidate
```

Replay samples use SB3's seeded NumPy RNG; guard replay sampling advances that
same NumPy RNG after the training batch. Evaluation does not advance learner
update keys, but **does advance the same policy-action key used by rollout**.
For exact historical evaluation environment seeding, the callback constructs
`jax.random.randint(jax.random.key(seed), (10_000_000,), 0, 2**30-1)` once, and
at evaluation t seeds the one-environment evaluation VecEnv with `seed_list[t]`.
Only the evaluation batch's first reset is reseeded; following episodes continue
that environment's RNG. Do not independently reseed every episode or restore the
policy key afterward when claiming identical historical RNG semantics.
Matching distributions in another framework does not give bitwise matching random
streams. Preserve these dependencies and record the framework/version differences.

## 11. Evaluation, diagnostics, saving and reporting

Online evaluation is stochastic, 10 episodes at t=1 and every 5000 environment
steps; returns are undiscounted environment reward sums. Include both terminated
and time-limit-truncated episodes. Primary score is each seed's mean of all 21
evaluations at 900000,905000,...,1000000, followed by the mean across four seeds.
Report sample SD across the four seed scores (ddof=1); it is not a confidence interval.

Independent final evaluation uses the saved post-update 1M actor, 50 episodes
per seed, environment seeds 1100000..1100049 and policy RNG seeds 1110000..1110049.
Reset both at each episode, stochastic actions, count all episodes. This uses a
different protocol from online evaluation and produced mean return 5397.09.

At diagnostic intervals (5000), retain per-state ESS statistics, Q mean/disagreement,
Q/T and -log q spread, teacher log density, policy σ and saturation, TD entropy
contribution, actor NLL, critic MSE, OT marginal residuals and guard acceptance/gap.
Raw importance ESS is computed before Sinkhorn's 1e-20 column-mass floor.
ESS does not trigger any adaptation in final v2. Guard rejection does not suppress
logging of the attempted proposal. Do not treat attempted-NLL improvement as an
accepted-actor improvement.

Save actor and critic TrainStates every 50000 steps and at completion, config,
source/version provenance and all evaluation arrays. Existing checkpoint files
include optimizer states and critic targets but **do not include the replay,
environment simulator state or all RNG states**. They support policy evaluation;
they are not exact mid-training-resume snapshots. A new implementation needing
exact resume must additionally serialize those states and all counters, without
claiming the historical checkpoint format already provides them.

## 12. Configuration traps and theory boundary

| Retained field | Actual meaning in final v2 |
|---|---|
| `transport_target_mode=argmax` | selection diagnostics; NLL uses the full plan |
| `proposal_clip=.5` | validated legacy field; no truncation in this proposal |
| `proposal_std_pretanh=.05` | alias of teacher-only floor, not fixed σ for all kernels |
| `minimum_source_ess=16` | inactive when adaptive density beta is false |
| `source_reference=uniform_action` | unused by this actor path; students come from policy |
| `v_min/v_max` | no scalar Q clipping |
| `bn_mode`, `bn_warmup`, `bn_momentum` | inactive because BN is disabled |
| `policy_tau=1` | target actor copy; TD uses current actor |

For exact old-policy soft Q define
`F_Q(pi,s)=E_pi Q(s,a)+T*H(pi(.|s)) = T*log Z - T*KL(pi || exp(Q/T)/Z)`.
If F does not decrease at every relevant state, then under the stated Bellman
boundedness/integrability assumptions soft policy improvement follows.
Exact Boltzmann extraction with exact distribution-preserving distillation is a
sufficient ideal case. Accurate Q alone does not turn arbitrary W2/NLL projection
into an improving update. This implementation has learned twin-Q, finite M/K,
Gaussian projection and a sampled replay-average filter; it does not certify
global or per-update improvement. Soft return and raw evaluation return also differ.
See [the complete theory note](THEORY.md).

## 13. Source mapping and independent review targets

| Specification | Implementation |
|---|---|
| Networks/inference | `optiq_dime/policy.py`, `models/critic.py`, `models/utils.py` |
| Density and proposal | `optiq_dime/semi_implicit.py` |
| Critic/actor/order | `optiq_dime/algorithm.py` |
| Sinkhorn | `optiq_dime/transport.py:sinkhorn` |
| NLL moments | `optiq_dime/distillation.py:conditional_ot_nll` |
| Guard | `optiq_dime/soft_improvement.py` |
| Adam/clipping | `optiq_dime/optimizers.py` |
| Replay and startup | `common/off_policy_algorithm.py`, `diffusion/dime.py`, SB3 2.1.0 |
| Online evaluation | `optiq_dime/evaluation.py`, `models/actor_critic_evaluation_callback.py` |
| Fresh final evaluation | `scripts/evaluate_v2_confirmation.py:evaluate_episodes` |

Mandatory port checks: joint-density/Jacobian identity; state isolation; same
sampling/density scales; generating-component inclusion; terminal vs timeout
bootstrap; twin-loss sum and global clipping; full-row NLL moment equality and
gradients; stopped Q/teacher/OT gradients; raw-cost Sinkhorn marginals; identical
old/new guard random draws; full optimizer rollback; correct default/archived
configs; actual Humanoid rollout/checkpoint round trip. [Review results](REVIEW.md).

IDAC estimator source: [Yue et al., NeurIPS 2020, Lemma 1](https://papers.neurips.cc/paper_files/paper/2020/file/4f20f7f5d2e7a1b640ebc8244428558c-Paper.pdf).
Soft policy improvement context: [SAC, section 4 and appendix B](https://arxiv.org/html/1801.01290v2).
The OT-NLL and sampled acceptance additions are this repository's construction.
