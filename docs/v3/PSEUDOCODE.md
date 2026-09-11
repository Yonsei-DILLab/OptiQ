# OptiQ v3 — 일반 TD + conditional-mixture OT NLL

구현 기준: `v3` 브랜치, `configs/v3/final.yaml`.
출발점: checked-K64 v2, commit `1aa7800`.
이 문서는 v3 학습과 실행을 재구현할 수 있는 의사코드다.
검증 범위와 실행 방법은 [README](../../README.md), [검증 기록](VALIDATION.md)을 참조한다.

## 1. v3의 정의

Critic은 **일반 discounted return**을 target twin-min TD로 학습한다.
Actor는 기존 continuous-latent 조건부 Gaussian 정책을 유지한다.
조건부 Gaussian mixture에서 teacher 후보를 뽑고, `beta=1` importance
weights와 16×64 OT를 이용한 전체 조건부 Gaussian NLL로 actor를 학습한다.
각 예정된 actor 업데이트를 그대로 적용한다.

정책 entropy는 TD target, actor의 추가 loss, 업데이트 수락/거절에 사용하지
않는다. IDAC entropy 평가와 entropy 진단 계산도 v3 학습 경로에서 실행하지
않는다. `g > 0` 검사, 별도 guard replay batch, optimizer rollback,
entropy coefficient 학습이 없다.

**유지하는 연산:** teacher의 `Q/T - log q`, entropic OT의 정규화 ε,
Gaussian NLL의 `+log sigma` 항이다. `-log q`는 proposal 밀도 보정이며,
`+log sigma`는 정규화된 Gaussian likelihood의 일부다. ε는 OT 배정을
부드럽게 만든다. 따라서 v3는 정책 entropy 보너스를 제거한 버전이며,
entropy와 관련된 모든 수학적 구조를 제거한 알고리즘이라는 뜻은 아니다.
Boltzmann teacher 자체는 여전히 온도 T로 분포를 부드럽게 한다.

## 2. 기본 상수와 shape

```python
B = 256                   # replay training batch
N = 16                    # 상태별 student latent/component 수
K = 64                    # 상태별 teacher 후보 수; N * 4
GAMMA = 0.99
TARGET_TAU = 0.005
TEACHER_T = 0.1            # teacher 가중치에만 사용
BETA = 1.0
BACKUP_ENTROPY_COEF = 0.0
ENTROPY_SAMPLES = 0
SOFT_GUARD = False
TEACHER_STD_FLOOR = 0.05   # pre-tanh, teacher proposal에만 적용
OT_EPSILON = 0.25
OT_ITERATIONS = 100
LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0
INITIAL_LOG_STD = log(0.5)
LR_ACTOR = LR_CRITIC = 3e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8            # sqrt 바깥, eps_root=0
GRAD_NORM_LIMIT = 2.0
WARMUP = ACTOR_START = 5_000
REPLAY_CAPACITY = TOTAL_ENV_STEPS = 1_000_000
UTD = POLICY_DELAY = 1
SEEDS = [0, 1, 2, 3]       # 독립 실험, vector env 수가 아님
DTYPE = float32
```

`O`는 관측 차원, `D`는 행동 차원이자 latent 차원이다. Humanoid-v4는
O=376, D=17이고, 다른 환경은 실제 space에서 읽는다.

| Tensor | Shape |
|---|---|
| states, replay actions | `[B,O]`, `[B,D]` |
| student z, mu, log_std, u, a | `[B,N,D]` |
| teacher v, b | `[B,K,D]` |
| teacher Q, log_q, weights | `[B,K]` |
| cost C, OT plan P, row probabilities R | `[B,N,K]` |
| reward, terminal, TD target | `[B]` |

OT는 상태마다 독립적으로 계산한다. 상태 간 수송은 없다.
Replay와 Q 입력, policy density, OT 비용의 행동 좌표는 `[-1,1]^D`이다.
환경에는 `low + (a+1)*(high-low)/2`로 변환해서 전달한다.
실제 critic 출력 `[2,B,1]`의 마지막 singleton 차원을 제거한 뒤 target을
계산해야 한다. `[B,1]`과 `[B]`를 잘못 broadcasting하지 않는다.

## 3. 네트워크와 실행 정책

```python
def ACTOR(theta, s, z):
    h = concat(s, z, axis=-1)
    for layer in theta.hidden:       # 256, 256, 256
        h = gelu_tanh_approx(linear(h, layer))
    mu = linear(h, theta.mu)
    ls = clip(linear(h, theta.log_std), -5, 1)
    return mu, ls

def Q(phi, s, a):
    h = concat(s, a, axis=-1)
    for layer in phi.hidden:         # 256, 256, 256
        h = gelu_tanh_approx(linear(h, layer))
    return linear(h, phi.output)[..., 0]  # raw scalar, no clipping/softmax

def SAMPLE_ACTION(theta, s, rng):
    z = rng.normal([len(s), D])
    eps = rng.normal([len(s), D])
    mu, ls = ACTOR(theta, s, z)
    return tanh(mu + exp(ls) * eps)
```

실제 정책은 `pi(a|s)=E_z[k_theta(a|s,z)]`, `z~N(0,I_D)`이다.
16은 학습할 때 뽑는 latent 수이며 고정 16개 codebook이 아니다.
환경 행동과 TD의 다음 행동 모두 위 함수로 생성한다. 행동 하나당 actor
forward 한 번이며 Q, OT, policy density 계산이 필요하지 않다.
Gaussian noise와 latent를 독립적으로 뽑는다.

Actor hidden weight는 fan-average uniform variance scaling(scale=1),
mu head는 같은 방식의 scale=1e-4, log_std head weight는 0,
bias는 log(.5)로 초기화한다. 나머지 bias는 0이다.
Critic은 Flax 기본 LeCun-normal weight, zero bias로 각각 독립 초기화한다.
LayerNorm, BatchNorm, dropout, 관측/보상 정규화가 없다.
GELU는 JAX/Flax 기본 tanh approximation이다.

## 4. 일반 TD critic 업데이트

```python
def UPDATE_CRITIC(theta, critics, targets, batch, rng):
    s, a, reward, s_next, terminal = batch

    with no_grad:
        a_next = SAMPLE_ACTION(theta, s_next, rng)   # 현재 actor
        next_q = minimum(Q(targets[0], s_next, a_next),
                         Q(targets[1], s_next, a_next))
        y = reward + GAMMA * (1 - terminal) * next_q

    def loss(phi):
        return mean((Q(phi[0], s, a) - y)**2) \
             + mean((Q(phi[1], s, a) - y)**2)

    grads = grad(loss)(critics.params)
    critics = ADAM_STEP(critics, clip_global_norm(grads, 2), lr=3e-4)
    targets = .995 * targets + .005 * critics.params
    return critics, targets
```

TD에는 `-T*log_pi`, entropy 추정치, 추가 TD smoothing noise가 들어가지
않는다. Teacher 온도를 바꿔도 고정 정책/critic/batch의 이 target 수식은
변하지 않는다. 전체 학습에서는 teacher 변경으로 정책이 달라질 수 있다.
True terminal에서는 `y=reward`; 시간 제한 truncation은 bootstrap한다.
Target action과 target Q를 통한 gradient는 차단한다.

## 5. Teacher 생성과 beta=1 밀도 보정

```python
def LOG_TANH_JACOBIAN(v):
    return sum(2 * (log(2) - v - softplus(-2*v)), axis=-1)

def LOG_PROPOSAL(v, proposal_mu, proposal_ls):
    # v [B,K,D]; component parameters [B,N,D]
    delta = (v[:,:,None,:] - proposal_mu[:,None,:,:]) \
            * exp(-proposal_ls[:,None,:,:])
    component_lp = sum(-.5*delta**2 - proposal_ls[:,None,:,:]
                       - .5*log(2*pi), axis=-1)
    return logsumexp(component_lp, axis=-1) - log(N) \
           - LOG_TANH_JACOBIAN(v)

def MAKE_TEACHER(theta_old, critics, s, rng):
    with no_grad:
        z = rng.normal([B,N,D])
        mu, ls = ACTOR(theta_old, broadcast_states(s,N), z)
        eps = rng.normal([B,N,D])
        a_student = tanh(mu + exp(ls)*eps)

        proposal_ls = maximum(ls, log(.05))
        indices = rng.integers(0, N, shape=[B,K])
        eta = rng.normal([B,K,D])
        v = gather_per_state(mu, indices) \
            + exp(gather_per_state(proposal_ls, indices))*eta
        b = tanh(v)
        log_q = LOG_PROPOSAL(v, mu, proposal_ls)

        q1 = Q(critics[0], broadcast_states(s,K), b)
        q2 = Q(critics[1], broadcast_states(s,K), b)
        logits = ((q1+q2)/2)/TEACHER_T - log_q  # beta = 1
        w = softmax(logits, axis=-1)

        C = sum((a_student[:,:,None,:] - b[:,None,:,:])**2, axis=-1)
        P = SINKHORN(C, w)
        R = P / maximum(sum(P, axis=-1, keepdims=True), 1e-20)
    return stop_gradient(z), stop_gradient(v), stop_gradient(R)
```

Teacher Q는 **업데이트된 live twin mean-Q**, TD는 **target twin min-Q**다.
Proposal은 실현된 행동을 중심으로 만든 KDE가 아니라 actor의 조건부
Gaussian 16개를 혼합한 분포다. 각 후보가 성분을 독립 균등 선택하므로
각 성분에서 정확히 네 개씩 뽑는 방식이 아니다.

Sampling과 log_q는 동일한 teacher-only std floor를 사용한다.
log_q는 선택된 한 성분의 밀도가 아니라 전체 16개 성분의 joint density다.
좌표별 log density를 합한 다음 성분 logsumexp를 적용한다.
이는 정의된 유한 proposal의 밀도이며 실제 continuous-latent 정책의
정확한 주변밀도라고 주장하지 않는다.

원래 pre-tanh v를 보관한다. `atanh(tanh(v))`로 복구하거나 tanh Jacobian을
생략하지 않는다. 학생 anchor, top-k, 후보 gradient ascent, hard cutoff가 없다.
가중치는 `exp(Q/T)/q`의 self-normalized importance weights이며, 후보가
유한하므로 정확한 Boltzmann 표본이라는 뜻은 아니다.

## 6. 상태별 16×64 Sinkhorn

```python
def SINKHORN(C, w):
    columns = clip(w, 1e-20, 1)
    columns /= sum(columns, axis=-1, keepdims=True)
    log_rows = full([B,N], -log(N))
    log_columns = log(columns)
    log_kernel = -C / .25
    log_u, log_v = zeros([B,N]), zeros([B,K])
    for _ in range(100):
        log_u = log_rows - logsumexp(log_kernel + log_v[:,None,:], axis=-1)
        log_v = log_columns - logsumexp(log_kernel + log_u[:,:,None], axis=-2)
    return exp(log_kernel + log_u[:,:,None] + log_v[:,None,:])
```

목적은 `<P,C> + epsilon*sum(P*(log(P)-1))`이고, 행 질량은 1/16,
열 질량은 보정된 teacher 가중치다. 비용은 normalized action 좌표의
**제곱거리 합**이다. 좌표 평균, 1/2 배율, 상태별 비용 정규화를 적용하지 않는다.
100회 반복은 행 marginal의 완전한 일치를 보장하지 않으므로, NLL에는
실제 행 합으로 정규화한 R을 사용하고 행/열 residual을 기록한다.

## 7. 전체 OT 행의 조건부 Gaussian NLL

```python
def UPDATE_ACTOR(actor, critics, s, rng):
    z, v, R = MAKE_TEACHER(actor.params, critics.params, s, rng)
    m = einsum('bnk,bkd->bnd', R, v)
    variance = maximum(einsum('bnk,bkd->bnd', R, v*v) - m*m, 0)
    m, variance = stop_gradient(m), stop_gradient(variance)

    def loss(theta):
        mu, ls = ACTOR(theta, broadcast_states(s,N), z)
        nll_coordinate = .5 * ((mu-m)**2 + variance) * exp(-2*ls) \
                         + ls + .5*log(2*pi)
        return mean(sum(nll_coordinate, axis=-1))  # sum D, mean B,N

    grads = grad(loss)(actor.params)
    return ADAM_STEP(actor, clip_global_norm(grads, 2), lr=3e-4)
```

정확히 다음 loss의 moment 구현이다.

```text
L_actor = -(1/(B*N)) sum_{b,i,j} R[b,i,j]
          * log Normal(v[b,j]; mu_theta(s[b],z[b,i]), diag(sigma_theta^2))
```

조건부는 `(s,z)`에 대한 조건부이다. OT는 tanh 이후 행동에서, NLL은
tanh 이전 공간에서 계산한다. Teacher tanh Jacobian은 고정 v에 대해
새 actor parameter와 무관하므로 이 NLL gradient에는 필요하지 않다.
`+log sigma`와 teacher variance를 모두 유지해 mu와 sigma를 학습한다.
Proposal, Q, weights, OT, teacher moments를 통한 gradient는 없다.
추가 actor Q-gradient, entropy-gradient, reverse-KL loss는 없다.

설정에 남은 `transport_target_mode=argmax`는 선택 진단에만 영향을 준다.
위 NLL은 argmax 행동 한 개 대신 전체 R을 사용한다.
Adam step과 parameter 업데이트는 매번 적용하며 수락 검사/rollback이 없다.

## 8. 전체 학습 루프

```python
initialize actor, independent twin critics, target critic copies, Adam states
initialize replay, environment, independent learner and rollout RNG streams
s = env.reset(seed)
for t in range(1, 1_000_001):
    if t <= 5_000:
        env_a = env.action_space.sample()
        a = NORMALIZE_ACTION(env_a)
    else:
        a = SAMPLE_ACTION(actor.params, s[None], rollout_rng)[0]
        env_a = UNSCALE_ACTION(a)
    s_next, reward, terminated, truncated, info = env.step(env_a)

    if t == 1 or t % 5_000 == 0:
        EVALUATE_CURRENT_POLICY(10 stochastic episodes)  # 현재 step 학습 이전

    replay.add(s, a, reward, s_next, terminal=terminated,
               timeout=truncated and not terminated)
    s = env.reset() if terminated or truncated else s_next
    if t > 5_000:
        batch = replay.sample_uniform_with_replacement(256)
        critics, targets = UPDATE_CRITIC(actor.params, critics, targets, batch, learner_rng)
        actor = UPDATE_ACTOR(actor, critics, batch.states, learner_rng)
        target_actor.params = actor.params   # 기존 checkpoint 호환용; v3 TD에서 미사용
    if t % 50_000 == 0:
        SAVE_TRAINSTATES(t)
SAVE_TRAINSTATES(1_000_000)
```

VecEnv를 사용할 때 replay에는 자동 reset 뒤 관측이 아니라 실제 terminal
observation을 넣는다. Teacher 생성 중 RNG를 고정해 loss/gradient 평가마다
새 표본을 다시 뽑지 않는다. Critic, teacher, rollout은 별도 RNG draw를 쓴다.
Guard용 RNG와 replay 샘플링은 없다. 따라서 같은 seed라도 v2와 v3의 전체
표본 경로가 bitwise 동일한 것은 아니다.

첫 학습은 step 5001이다. 정상적인 기본 1M 실행에서는 critic과 actor가
각각 995,000번 업데이트된다. Adam은 bias correction을 사용하고 gradient
clipping을 Adam 이전에 적용한다. Critic 두 네트워크의 gradient norm은
하나로 합쳐 clipping하며 actor는 별도 clipping한다.

## 9. 평가, 기록, 사용 범위

기본 환경은 Gymnasium 0.29.1 / MuJoCo 2.3.7의 Humanoid-v4다.
환경/seed/evaluation callback과 기본 hyperparameter는 v2를 계승한다.
평가는 확률적이며 latent와 Gaussian noise를 모두 샘플링한다. 평가가
rollout용 policy RNG를 진행시키는 기존 동작도 유지한다.
5K 간격 10 episode 평가, 50K 간격 checkpoint, 5K 간격 상세 진단을 기록한다.
900K–1M의 21개 평가 평균을 seed별로 계산한 뒤 seed 간 비교한다.

기록: 일반 Q/TD loss, NLL, teacher 온도, ESS, twin disagreement,
조건부 sigma, latent mean variance fraction, 행동 포화도, OT residual,
실제 actor Adam step. `ent_coef`, `backup_entropy_term`은 항상 0이다.
`policy_entropy_lower`, `backup_entropy_lower`, guard gap은 계산/기록하지 않는다.

W&B: 기존 `OptiQ/optiq_mujoco_v2_confirmation` 프로젝트의 별도 v3 그룹.
출력: `../optiq-experiments/v3_td/outputs/` 아래 매 실행 새 ID/디렉터리.
Checkpoint에는 optimizer를 포함한 actor/critic TrainState가 있으나 replay,
환경 상태, 모든 RNG는 없으므로 정확한 중간 학습 재개 checkpoint는 아니다.
코드 검증과 짧은 환경 실행은 성능 실험이 아니다. 요청된 본 실험은 seed별
1M까지 실행하며 성능에 따른 조기 종료를 적용하지 않는다.

v3는 일반 Q 평가와 Boltzmann teacher/OT projection을 조합한 알고리즘이다.
NLL 감소가 일반 return 증가나 단조 policy improvement를 보장하지 않는다.

## 10. 코드 대응

| 단계 | 구현 |
|---|---|
| 기본 설정 | `configs/v3/final.yaml` |
| 정책/행동 샘플 | `optiq_dime/policy.py:OptiQPolicy.sample_action` |
| 일반 TD/학습 순서 | `optiq_dime/algorithm.py:update_critic`, `_train` |
| Teacher mixture | `optiq_dime/semi_implicit.py:ConditionalGaussianProposal` |
| Weights/OT 연결 | `optiq_dime/algorithm.py:update_actor` |
| Sinkhorn | `optiq_dime/transport.py:sinkhorn` |
| 조건부 NLL | `optiq_dime/distillation.py:conditional_ot_nll` |
| 실행/설정 검증 | `run_optiq_dime.py`, `scripts/verify_v3.py`, `scripts/run_v3.sh` |
| 검증 | `tests/test_v3.py` |
