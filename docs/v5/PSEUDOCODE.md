# OptiQ v5 — mean-action OT + conditional Gaussian NLL + 일반 TD

구현: `configs/v5/final.yaml` (alias `mujoco_v5`), `optiq_dime/algorithm.py`.
v4와의 차이와 근거는 [변경 설명](CHANGES_KO.md), 검사 범위는 [검증 기록](VALIDATION.md)을 참조한다.

## 1. v5의 정의

정책은 continuous-latent 조건부 Gaussian이다.

```text
z ~ Normal(0, I_D)
(mu, log_sigma) = G_theta(s, z)
u = mu + sigma * epsilon,  epsilon ~ Normal(0, I_D)
a = tanh(u)
```

환경 수집과 TD next action은 위 전체 Gaussian 정책을 사용한다.
**OT student 위치만 `tanh(mu(s,z))`를 사용한다.** Teacher 후보는 여전히
mu와 sigma를 가진 Gaussian mixture에서 뽑으며, 배정 후 NLL은 두 head를 모두 학습한다.

정책 entropy bonus, soft-value guard, optimizer rollback은 없다. Teacher의
`Q/T - log q`, entropic OT 정규화, NLL의 `+log_sigma`는 각각 밀도 보정·수송
정규화·Gaussian likelihood의 일부이며 그대로 유지한다.

## 2. 기본 설정과 shape

```python
B = 256                         # replay batch
M = 16                          # 매 상태마다 새로 뽑는 student latent 수
K = M * 4                       # teacher 후보 64개
ACTOR_WIDTHS = CRITIC_WIDTHS = [256, 256]
T = 0.25                        # 고정 teacher 온도; annealing 없음
BETA = 1.0
LAMBDA_OT = 0.25                 # OT 정규화: Gaussian epsilon과 별개
SINKHORN_ITERATIONS = 100
TEACHER_STD_FLOOR = 0.05
LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0
INITIAL_SIGMA = 0.5
GAMMA = 0.99
TARGET_TAU = 0.005
LR_ACTOR = LR_CRITIC = 3e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
GLOBAL_GRAD_CLIP = 2.0
WARMUP = ACTOR_START = 5_000
REPLAY_CAPACITY = TOTAL_STEPS = 1_000_000
UTD = POLICY_DELAY = 1
UNIFORM_REPLACEMENT_PROBABILITY = 0.0  # warmup 이후 추가 uniform 탐색 없음
EVAL_INTERVAL = DIAGNOSTIC_INTERVAL = 5_000
EVAL_EPISODES_PER_MODE = 10
CHECKPOINT_INTERVAL = 50_000
DTYPE = float32
```

Warmup 5K의 원래 uniform action 수집은 유지한다. `uniform=0`은 warmup 이후
정책 행동을 추가 uniform 행동으로 대체하지 않는다는 뜻이다.

`O`는 관측 차원, `D`는 행동 및 latent 차원이다. Ant-v4는 O=27, D=8이며,
다른 환경에서는 space에서 읽는다. Default benchmark는 v4에서 상속한 Humanoid-v4다.
Ant 실험은 `benchmark=ant`로 명시한다. M=16은 고정 codebook 크기가 아니다.

| Tensor | Shape |
|---|---|
| states, replay actions | `[B,O]`, `[B,D]` |
| student z, mu, log_sigma, epsilon | `[B,M,D]` |
| teacher pre-tanh v, action b | `[B,K,D]` |
| teacher Q, log_q, weights | `[B,K]` |
| OT cost C, plan P, row probabilities R | `[B,M,K]` |

Q와 OT의 행동 좌표는 `[-1,1]^D`이며 env.step에는 action bounds로 unscale한다.
OT는 각 상태에서 독립적으로 계산한다. 관측·보상 정규화, LayerNorm,
BatchNorm, dropout은 기본 설정에서 사용하지 않는다.

## 3. Actor, critic, 행동 생성

```python
def ACTOR(theta, s, z):
    h = concat(s, z)
    for layer in theta.hidden:      # 서로 다른 256, 256 두 layer
        h = gelu_tanh_approx(linear(h, layer))
    mu = linear(h, theta.mu_head)
    log_sigma = clip(linear(h, theta.sigma_head), -5, 1)
    return mu, log_sigma

def SAMPLE_ACTION(theta, s, key):
    z_key, eps_key = split(key)
    z = normal(z_key, [len(s), D])
    eps = normal(eps_key, [len(s), D])
    mu, ls = ACTOR(theta, s, z)
    return tanh(mu + exp(ls) * eps)

def Q(phi, s, a):
    h = concat(s, a)
    for layer in phi.hidden:         # 256, 256
        h = gelu_tanh_approx(linear(h, layer))
    return linear(h, phi.output)[..., 0]
```

위 actor hidden loop의 두 layer는 서로 다른 파라미터를 사용한다.
Actor hidden weight는 fan-average uniform variance scaling(scale=1), mu head는
scale=1e-4, sigma head weight는 0, bias는 log(.5)로 초기화한다. Critic 두 개는
독립적인 Flax LeCun-normal 초기화를 사용한다. 초기화를 변경한 실험이 아니다.

## 4. 일반 TD backup

```python
def UPDATE_CRITIC(actor, critics, target_critics, batch, key):
    s, a, r, s_next, terminal = batch
    next_key, action_key, unused_noise_key, dropout_key_target, dropout_key_current, unused_key = split(key, 6)
    with no_grad:
        a_next = SAMPLE_ACTION(actor.params, s_next, action_key)
        q_next = minimum(Q(target_critics[0], s_next, a_next),
                         Q(target_critics[1], s_next, a_next))
        y = r + GAMMA * (1 - terminal) * q_next
    def loss(phi):
        return mean((Q(phi[0], s, a) - y)**2) \
             + mean((Q(phi[1], s, a) - y)**2)
    critics = adam_step(critics, clip_global_norm(grad(loss)(critics.params), 2))
    target_critics = .995 * target_critics + .005 * critics.params
    return critics, target_critics, next_key
```

Next action은 현재 actor의 **mu + sigma*epsilon**에서 나온다. `tanh(mu)`로
바뀌는 곳은 OT student 비용뿐이다. TD에 `-T log pi`, entropy 추정,
별도 smoothing noise를 더하지 않는다. True terminal만 bootstrap을 끄고,
시간 제한 truncation은 실제 마지막 관측에서 bootstrap한다.

## 5. Gaussian teacher 생성과 beta=1 중요도 가중치

```python
def LOG_PROPOSAL(v, mu, proposal_ls):
    delta = (v[:,:,None,:] - mu[:,None,:,:]) * exp(-proposal_ls[:,None,:,:])
    component_lp = sum(-.5*delta**2 - proposal_ls[:,None,:,:]
                       - .5*log(2*pi), axis=-1)
    pretanh_lp = logsumexp(component_lp, axis=-1) - log(M)
    tanh_log_jac = sum(2*(log(2)-v-softplus(-2*v)), axis=-1)
    return pretanh_lp - tanh_log_jac

def MAKE_ASSIGNMENT(theta_old, critics, s, key):
    next_key, latent_key, proposal_key, dropout_key = split(key, 4)
    z_key, eps_key = split(latent_key)
    z = normal(z_key, [B,M,D])
    with no_grad:
        mu, ls = ACTOR(theta_old, repeat_states(s,M), z)
        eps = normal(eps_key, [B,M,D])
        sampled_student = tanh(mu + exp(ls)*eps)  # 기존 draw/진단 유지
        student_position = tanh(mu)              # v5의 변경점

        proposal_ls = maximum(ls, log(.05))      # teacher에만 std floor
        component_key, noise_key = split(proposal_key)
        indices = randint(component_key, 0, M, shape=[B,K])
        eta = normal(noise_key, [B,K,D])
        v = gather(mu, indices) + exp(gather(proposal_ls, indices))*eta
        b = tanh(v)
        log_q = LOG_PROPOSAL(v, mu, proposal_ls)

        q1 = Q(critics[0], repeat_states(s,K), b)
        q2 = Q(critics[1], repeat_states(s,K), b)
        w = softmax(((q1+q2)/2)/T - log_q, axis=-1)

        C = sum((student_position[:,:,None,:] - b[:,None,:,:])**2, axis=-1)
        P = SINKHORN(C, w)
        R = P / maximum(sum(P, axis=-1, keepdims=True), 1e-20)
    return z, stop_gradient(v), stop_gradient(R), next_key
```

Teacher는 방금 업데이트한 live twin **mean-Q**, TD는 target twin **min-Q**를 쓴다.
`exact` proposal sampling은 각 후보가 component를 독립 균등 선택하는 방식이다.
M개 component에서 정확히 4개씩 뽑는 stratified sampling과 다르다.

**Sigma는 teacher 후보와 proposal density 양쪽에 포함된다.** Sampling과
log_q 모두 같은 teacher-only floor를 사용한다. log_q는 좌표별 Gaussian을
곱한 joint density에 component logsumexp를 적용한 값이다. 원래 pre-tanh v를
보관하며 `atanh(tanh(v))`로 복구하지 않는다.

가중치는 `exp(Q/T)/q(a|s)`의 self-normalized importance weights다. 이 q는
뽑힌 M개 conditional Gaussian으로 정의한 유한 proposal의 정확한 밀도이며,
continuous-z 정책 주변밀도의 정확한 계산이나 정확한 Boltzmann sampling을 뜻하지 않는다.

## 6. Mean-action entropic OT

```python
def SINKHORN(C, w):
    columns = clip(w, 1e-20, 1)
    columns /= sum(columns, axis=-1, keepdims=True)
    log_rows = full([B,M], -log(M))
    log_columns = log(columns)
    log_kernel = -C / LAMBDA_OT
    log_u, log_v = zeros([B,M]), zeros([B,K])
    for _ in range(100):
        log_u = log_rows - logsumexp(log_kernel + log_v[:,None,:], axis=-1)
        log_v = log_columns - logsumexp(log_kernel + log_u[:,:,None], axis=-2)
    return exp(log_kernel + log_u[:,:,None] + log_v[:,None,:])
```

목적은 `<P,C> + LAMBDA_OT * sum(P*(log(P)-1))`, 행 marginal은 1/M,
열 marginal은 teacher weights다. C는 action 좌표의 제곱거리 **합**으로,
좌표 평균이나 상태별 비용 정규화를 사용하지 않는다. 유한 iteration 이후
실제 행 합으로 R을 정규화하고 행/열 오차를 기록한다.

Student sigma가 비용에 직접 들어가지는 않지만, teacher 샘플과 밀도에는
sigma가 있으므로 **OT 전체가 sigma와 무관한 것은 아니다.** 같은 mu에 다른
sigma를 가진 student 둘은 같은 비용 행을 가지며 배정에서 구분되지 않는다.

## 7. 배정 후 mu와 sigma의 NLL 학습

```python
def UPDATE_ACTOR(actor, critics, s, key):
    z, v, R, key = MAKE_ASSIGNMENT(actor.params, critics.params, s, key)
    m = stop_gradient(einsum('bmk,bkd->bmd', R, v))
    V = stop_gradient(maximum(einsum('bmk,bkd->bmd', R, v*v) - m*m, 0))

    def loss(theta):
        mu, ls = ACTOR(theta, repeat_states(s,M), z)
        per_coordinate = .5*((mu-m)**2 + V)*exp(-2*ls) + ls + .5*log(2*pi)
        return mean(sum(per_coordinate, axis=-1))  # sum D; mean B,M

    actor = adam_step(actor, clip_global_norm(grad(loss)(actor.params), 2))
    return actor, key
```

이는 `-mean_{s,i} sum_j R_ij log Normal(v_j; mu_i, diag(sigma_i^2))`와 같다.
Mu는 배정된 pre-tanh 샘플의 평균, sigma는 그 주변 잔차 분산을 학습한다.
Teacher component의 sigma를 학생에게 그대로 복사하는 방식이 아니다.
제약 없이 정확히 최소화할 경우 `mu_i=m_i`, `sigma_i^2=V_i`다. 실제 신경망은
파라미터 공유, log-std 범위, 유한 optimizer step 때문에 이 해와 다를 수 있다.

전체 R을 사용하며 `transport_target_mode=argmax`는 선택 진단에만 영향을 준다.
Teacher, Q, weights, OT plan과 teacher moments를 통한 gradient는 차단한다.
Actor gradient는 위 NLL의 mu·log_sigma로 흐르고 매번 적용된다.

## 8. 전체 루프와 두 평가

```python
initialize actor, independent twin critics, target critic copies, Adam, replay
initialize independent collection, learning and evaluation RNG streams
s = env.reset(seed)
for step in range(1, TOTAL_STEPS+1):
    if step <= WARMUP:
        a = uniform_action()
    else:
        collection_key, action_key = split(collection_key)
        a = SAMPLE_ACTION(actor.params, s[None], action_key)[0]
    s_next, reward, terminated, truncated = env.step(unscale(a))
    if step == 1 or step % 5000 == 0:
        EVALUATE_BOTH_MODES(actor)                 # 해당 step의 학습 전
    replay.add(s, a, reward, real_terminal_observation(s_next), terminated, truncated)
    s = reset_if_done(s_next)
    if step > WARMUP:
        batch = replay.sample_uniform(256)
        critics, targets, learning_key = UPDATE_CRITIC(actor, critics, targets, batch, learning_key)
        actor, learning_key = UPDATE_ACTOR(actor, critics, batch.states, learning_key)
    if step == 5001 or step % 50000 == 0:
        SAVE_TRAINSTATES(step)

def EVALUATE_BOTH_MODES(actor):
    for mode in ['zero_z', 'stochastic_z']:
        for episode in range(10):
            s = eval_env.reset(paired_environment_seed[episode])
            rng = isolated_rng(paired_policy_seed[episode])
            while not done:
                z = zeros(D) if mode == 'zero_z' else rng.normal(D)
                mu, _ = ACTOR(actor.params, s, z)
                a = tanh(mu)                     # 두 모드 모두 epsilon=0
                s, reward, done = eval_env.step(unscale(a))
```

기본 1M 실행에서 actor/critic 각각 995,000회 업데이트한다. 저장되는 target
actor는 기존 checkpoint 호환용이며 v5 TD next action은 live actor를 사용한다.
정기 평가는 해당 env step의 학습 전에, 정기 checkpoint는 학습 후에 수행하는
기존 v4 순서를 유지한다. Evaluation은 collection/learner RNG를 진행시키지 않는다.

Sampled-z 평가에서는 행동마다 z를 다시 뽑는다. `tanh(mu(s,0))`는 고정 latent의
대표 행동이지 전체 mixture의 평균이나 MAP라는 보장은 없다. `tanh(mu)` 역시
Gaussian epsilon까지 적분한 tanh 행동의 기대값과 일반적으로 다르다.

## 9. 실행·기록·검증 범위

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python bash scripts/run_v5.sh 0 --check benchmark=ant
# 학습이 요청된 경우에만 실행; GPU worker는 supervisor로 관리한다.
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python bash scripts/run_v5.sh 0 benchmark=ant
# 기본 entrypoint 역시 mujoco_v5다.
python run_optiq_dime.py --config-name=mujoco_v5 benchmark=ant seed=0
```

W&B는 기존 사용자 지정 프로젝트 `OptiQ/v4-test`의 v5 group/run 이름을 쓴다.
출력은 `../optiq-experiments/v5/outputs` 아래 새 run ID/디렉터리다.
두 평가의 episode reward·환경 seed·정책 seed를 별도 npz에 저장한다.
`eval/mean_reward`는 zero-z 호환 alias이며 두 모드 중 높은 점수를 고르는 지표가 아니다.

Teacher ESS, NLL, sigma, latent mean variance fraction, OT residual을 유지한다.
`policy_spread_l2`는 기존 Gaussian sampled action 진단이므로 mean-only spread로
해석하지 않는다. Sigma가 없는 비용의 `ot_cost_mean`도 v4와 같은 분포의 비용은 아니다.

완료된 Ant 대조는 후반 성능 개선과 초기 평균 분화 증가를 보였지만, 모든 seed의
학습 가속이나 후반 mean 다양성 유지, 다른 환경의 성능 개선을 증명하지 않았다.
NLL 감소가 return의 단조 증가 또는 정확한 Q-reference marginal fit을 보장하지 않는다.
자세한 결과는 [4시드 분석](../v4/MEAN_OT_RESULTS_KO.md)에 보존했다.
