# OptiQ v7: OT로 분배한 조건부 Boltzmann 정책 추출

기준: 2026-09-19, `/root/OptiQ-v7`, `v7` 브랜치. 현재 기본 actor objective는
`ot_conditional_sac`다. 사용자의 선택에 따라 **조건부 entropy와 OT 배정항을 쓰는
이전 방식으로 복원**했다. 목표는 Q가 정의하는 Boltzmann 분포를 latent별로 분배하여
추출하는 것이며, actor loss를 marginal SAC와 동일하게 만드는 것이 목표는 아니다.

전체 4096-component marginal density를 계산하던 변경은 별도 비교 구현으로
[복원 전 archive](/root/optiq-archives/v7-before-conditional-restore-20260919T232802Z)에
보존한다. 기존 ε 비교와 conditional GMM40 결과는 현재 계열의 실험 기록이다.
복원 범위와 검증은 [CONDITIONAL_RESTORE_KO.md](CONDITIONAL_RESTORE_KO.md)에 기록한다.

```text
고정 normal latent 적분점 H=4096 ─────────────────────────┐
                                                        │
현재 actor → fresh teacher latent 256개                  │
           → 각 Gaussian에서 행동 하나                  │
           → W ∝ exp(Q/T)/q                             │
           → 16개 importance 재표집                     │
           → persistent potential로 4096×16 OT 배정 ←───┘
           → 각 teacher occurrence에서 latent 하나 선택
           → 선택된 16 latent의 actor Gaussian에서 새 행동
           → source correction × [T log π_i(a) − Q(a) − T log Pr(i|a,s)]

Critic: fresh 16-component marginal density 근사를 사용하는 soft TD
```

4096은 **OT의 latent 적분점 수**다. 현재 actor loss를 위해 4096개 Gaussian 출력의
전체 mixture density를 계산하지 않는다. Teacher의 256개 출력과 학습에 선택된
16개 Gaussian 출력을 계산하며, 새 행동의 OT 배정 query는 4096개 적분점을 사용한다.

## 1. 표기와 정책

기존 teacher 행동 $b_j$, normalized importance weight $W_j$, coupling $P_{ij}$,
행별 조건부 $R_{ij}$를 유지한다. [NOTATION_KO.md](NOTATION_KO.md)가 공통 표기 기준이다.

| 기호 | 의미 | 기본값 |
|---|---|---:|
| $B$ | replay minibatch 또는 GMM Monte Carlo lane 수 | 256 |
| $z_i$, $H$ | standard-normal latent 적분점과 수 | 4096 |
| $\pi_i(a\mid s)$ | 동일한 actor MLP가 $z_i$에서 정의하는 tanh-Gaussian | — |
| $\pi_{\theta,H}$ | $H^{-1}\sum_i\pi_i$, 정책의 유한 적분점 근사 | 이론·진단용 |
| $M$, $M_{\rm prop}$ | 원래 teacher 후보 수와 proposal Gaussian 수 | 둘 다 256 |
| $b_j$, $v_j$ | 원래 teacher 행동과 저장된 pre-tanh 좌표 | $b_j=\tanh v_j$ |
| $W_j$ | `softmax(Q/T − log q)` | 합 1 |
| $K$, $\tilde b_j$, $\tilde v_j$ | importance 재표집한 teacher occurrence와 좌표 | 16, 중복 허용 |
| $P_{ij}$ | source i와 teacher occurrence j의 OT coupling | $H\times K$ |
| $R_{ij}$ | $P_{ij}/\sum_kP_{ik}$, teacher 방향 행 조건부 | $\sum_jR_{ij}=1$ |
| $\Pr(i\mid a,s)$ | latent 방향 연속 OT 배정 확률 | $\sum_i\Pr(i\mid a,s)=1$ |
| $\sum_jP_{ij}$ | 현재 유한 teacher에서 측정한 source 행 질량 | empirical |
| $\bar P_i$ | 실제 Boltzmann 분포에 대한 source 질량 | population |
| $L$ | soft-TD marginal density 근사의 fresh component 수 | 16 |
| $T=\alpha$ | teacher·actor·soft TD의 공통 고정 온도 | RL .25 / GMM 1 |
| $\varepsilon_{\rm OT}$ | entropic OT regularization | .1 |

Actor는 기존 shared MLP와 두 출력 head를 사용한다.

$$
(\mu_\theta(s,z),\ell_\theta(s,z))=g_\theta(s,z),\quad
\sigma_\theta(s,z)=e^{\ell_\theta(s,z)},\quad
u=\mu_\theta(s,z)+\sigma_\theta(s,z)\odot\xi,\quad
\xi\sim\mathcal N(0,I),\quad a=\tanh u.
$$

하나의 z에는 diagonal Gaussian 하나가 대응하지만 전체 정책은

$$
\pi_\theta(a\mid s)=\int p_0(z)\pi_\theta(a\mid s,z)\,dz,
\qquad
\pi_{\theta,H}(a\mid s)=\frac1H\sum_i\pi_i(a\mid s)
$$

이므로 multimodal일 수 있다. 적분점 가중치 $1/H$는 Gaussian 좌표 밀도
$p_0(z_i)$와 다르다. 모든 Gaussian은 같은 MLP를 공유한다.

Actor의 초기 σ는 .5, logσ 범위는 `[-5,1]`이다. Teacher proposal의 .05 하한은
actor 행동 생성·조건부 log-density·TD density에 적용하지 않는다.

$$
\log\pi_i(a\mid s)
=\log\mathcal N(u;\mu_i,\operatorname{diag}\sigma_i^2)
-\sum_d\log(1-\tanh^2u_d).
$$

물리 좌표 $x=c\tanh u$의 밀도에는 $-d\log c$를 추가한다. GMM에서 c=40이다.
구현은 안정적인 tanh log-Jacobian 식을 사용한다.

환경 수집과 stochastic-z 평가는 fresh $z\sim\mathcal N(0,I)$를 사용하며,
zero-z 평가는 z=0이다. RL의 두 평가 모드는 Gaussian noise를 0으로 둔다.
Boltzmann 복원 명제는 noise-free 평가가 아니라 확률적 정책에 대한 것이며,
유한 적분점 정책과 연속 latent 실행 정책의 차이도 별도로 남는다.

## 2. Teacher: Boltzmann importance correction과 256→16 재표집

각 상태·업데이트마다 fresh normal latent 256개를 뽑아 업데이트 전 actor로
Gaussian 256개를 만든다. Proposal sampling과 density 양쪽에
$\tilde\sigma_j=\max(\sigma_j,.05)$를 사용한다.

$$
q_u(v\mid s)=\frac1M\sum_{j=1}^M
\mathcal N(v;\mu_j,\operatorname{diag}\tilde\sigma_j^2),\qquad M=256.
$$

각 Gaussian에서 후보 하나씩 생성하는 stratified proposal이다. 후보 $b_j$의
importance denominator는 생성 Gaussian 하나가 아니라 **전체 256-component q**다.
Tanh Jacobian과 필요한 물리 scale Jacobian을 포함한다.

$$
\pi_B(a\mid s)=\frac{e^{Q(s,a)/T}}{Z(s)},\qquad
W_j=\frac{\exp(Q(s,b_j)/T-\log q(b_j\mid s))}
{\sum_{k=1}^M\exp(Q(s,b_k)/T-\log q(b_k\mid s))}.
$$

W로 K=16개 occurrence를 stratified importance resampling한다. 중복을 유지하고
OT 열 질량은 각각 1/K다. **W는 빈도에 이미 반영되었으므로 다시 곱하지 않는다.**
Teacher latent·출력·noise·행동·Q값·W·재표집은 매 step 새로 만들고 actor gradient에서 분리한다.

Proposal support와 적분 가능성 조건에서 self-normalized importance sampling은
Boltzmann 적분을 근사한다. 유한 256개에서 정확하거나 불편하다고 가정하지 않는다.
Teacher를 고정해 재사용하거나 원래 후보 256개 모두를 OT에 넣는 방식이 아니다.

## 3. Latent OT와 persistent semi-dual

Seed에서 만든 iid normal 적분점 Z를 모든 상태·업데이트에 재사용한다.
고정하는 것은 좌표이며 teacher와 dual parameter는 계속 변한다.

$$
C_{ij}=\|z_i-\tilde v_j\|^2.
$$

비용은 raw squared latent–pre-tanh distance다. Actor 출력 위치·student sigma·배치별
cost normalization은 사용하지 않는다. Gaussian covariance를 비교하는 OT도 아니다.

이상적 entropic OT의 source·teacher marginal은 각각 1/H와 1/K다.
현재는 persistent state-conditioned dual MLP $f_\phi(s)$로 다음 semi-dual을 추적한다.

$$
D_s(f)=\frac{\varepsilon_{\rm OT}}K\sum_j
\log\left[\frac1H\sum_i e^{(f_i-C_{ij})/\varepsilon_{\rm OT}}\right]
-\frac1H\sum_i f_i,
\qquad
\frac{\partial D_s}{\partial f_i}=\sum_jP_{ij}-\frac1H.
$$

Dual은 ReLU 256×2→H, zero output initialization, 평균을 빼는 gauge, Adam 1e-4다.
매 actor step에서 dual도 한 번 갱신하고 parameter·Adam moments를 유지한다.
매 batch를 처음부터 새로 풀지 않는다. `fresh_sinkhorn` 100회는 별도 대조군이다.

연속 배정 함수와 coupling은

$$
\Pr(i\mid a,s)=\operatorname{softmax}_{i}
\frac{f_i(s)-\|z_i-u(a)\|^2}{\varepsilon_{\rm OT}},\qquad
P_{ij}=\frac1K\Pr(i\mid\tilde b_j,s),\qquad
R_{ij}=\frac{P_{ij}}{\sum_kP_{ik}}.
$$

Pr은 latent 방향, R은 teacher 방향으로 정규화한다. 두 함수를 같은 기호로 부르지 않는다.
Teacher 열 질량은 항상 1/K이지만 각 batch의 행 질량이 1/H와 맞는다는 보장은 없다.

Actor와 dual gradient는 같은 **업데이트 전 f**를 사용한다. GMM에서는 상수 상태 하나의
f를 B개 lane에 공유하고 batch 평균 행 질량 gradient를 전달한다. RL에서는 각 replay
상태의 f(s)를 사용한다. Replay 항목별 벡터를 저장하는 방식은 아니다.

## 4. 조건부 Boltzmann 목표와 복원 조건

상태와 Q, 배정 함수를 고정하고 모집단 source 질량을

$$
\bar P_i(s)=\int\pi_B(a\mid s)\Pr(i\mid a,s)\,da
$$

로 정의한다. 이는 유한 16개 teacher에서 측정한 $\sum_jP_{ij}$와 다르다.
Latent i의 정규화된 조건부 목표는

$$
t_i(a\mid s)=\frac{\pi_B(a\mid s)\Pr(i\mid a,s)}{\bar P_i(s)}.
$$

현재 actor는 $\pi_i$를 이 목표에 맞춘다. 다음 두 조건은 Boltzmann 복원의 충분조건이다.

1. 실제 Boltzmann 목표에서 OT source 질량이 prior와 일치: $\bar P_i=1/H$.
2. 각 actor Gaussian conditional이 배정된 목표에 적합: $\pi_i=t_i$.

그러면

$$
\boxed{
\pi_{\theta,H}(a\mid s)=\frac1H\sum_i\pi_i(a\mid s)
=\pi_B(a\mid s)\sum_i\Pr(i\mid a,s)=\pi_B(a\mid s).
}
$$

OT는 이미 Q와 T로 정의된 Boltzmann 행동 질량을 latent별로 나눈다.
정답 mode label이 필요하지 않지만 **한 latent에 반드시 한 mode만 배정한다는 보장도 없다.**
$t_i$가 multimodal이면 하나의 diagonal Gaussian이 정확히 표현하지 못할 수 있다.

불완전한 적합과 질량 오차를 분리하면

$$
\mathrm{TV}(\pi_{\theta,H},\pi_B)
\le\frac1H\sum_i\mathrm{TV}(\pi_i,t_i)
+\mathrm{TV}(\mathrm{Unif}_H,\bar P).
$$

대응하는 KL 상한은 적절한 support·유한성 조건에서

$$
\mathrm{KL}(\pi_{\theta,H}\|\pi_B)
\le\mathrm{KL}(\mathrm{Unif}_H\|\bar P)
+\frac1H\sum_i\mathrm{KL}(\pi_i\|t_i)
$$

다. Population balance·조건부 적합·유한 적분점 일반화는 서로 다른 오차다.
유한 표본과 신경망 SGD가 이 모든 조건을 달성하거나 전역 수렴한다는 주장은 하지 않는다.

## 5. 실제 actor 16쌍 학습과 gradient

재표집 teacher occurrence j마다 latent index를 하나 선택하고 새 Gaussian noise로
actor 행동을 생성한다.

$$
i_j\sim\Pr(\cdot\mid\tilde b_j,s),\qquad
u_j=\mu_\theta(s,z_{i_j})+\sigma_\theta(s,z_{i_j})\odot\xi_j,
\quad \xi_j\sim\mathcal N(0,I),\quad a_j=\tanh u_j.
$$

평균 source 선택 확률은 $\sum_jP_{ij}$다. 목표 prior 1/H로 보정한 loss는

$$
\boxed{
\widehat J_{\rm OT}=\frac1{BK}\sum_{s\in\mathcal B}\sum_{j=1}^K
\operatorname{sg}\left[\frac{1/H}{\sum_kP_{i_jk}}\right]
\left[T\log\pi_{i_j}(a_j\mid s)-Q(s,a_j)
-T\log\Pr(i_j\mid a_j,s)\right].
}
$$

Teacher 좌표로 NLL 회귀하는 것이 아니라 선택된 Gaussian이 새로 생성한 행동에서
Q와 배정 확률을 평가한다. Actor의 μ·σ·shared trunk는 같은 loss로 동시에 학습한다.

OT potential·적분점·categorical index·source 보정계수·critic parameter는 고정하되,
**새 행동 입력에 대한 Q와 Pr의 gradient는 유지**한다.

$$
\nabla_u\log\Pr(i\mid\tanh u,s)
=\frac2{\varepsilon_{\rm OT}}
\left(z_i-\sum_l\Pr(l\mid\tanh u,s)z_l\right).
$$

Pr 전체를 detach하면 조건부 목표를 정의하는 배정 힘이 사라진다.
Density는 선택된 Gaussian의 actual σ를 사용하고 teacher floor를 넣지 않는다.

고정 map에서 source i의 새 noise에 대해 평균한 local gradient를 $G_i$라 하면

$$
\frac1K\sum_j\sum_i\Pr(i\mid\tilde b_j,s)
\frac{1/H}{\sum_kP_{ik}}G_i=\frac1H\sum_iG_i.
$$

이는 source 확률이 양수인 경우의 기대 gradient 항등식이다. 기대 actor objective는

$$
J_{\rm OT}=\frac1H\sum_i\mathbb E_{\pi_i}
[T\log\pi_i-Q-T\log\Pr(i\mid a,s)]
=\frac TH\sum_i\mathrm{KL}(\pi_i\|t_i)+\text{actor에 무관한 상수}
$$

가 된다. Source 보정은 **teacher W를 다시 적용하는 것이 아니며** 실제 16쌍 gradient나
Adam step을 균등 표집과 같게 만드는 것도 아니다. 또한 모집단 질량 $\bar P_i$나
조건부 fitting을 직접 보정하지 않는다.

W는 teacher와 persistent potential을 통해 조건부 목표에 영향을 준다. f를 고정한
한 step에서는 source 보정 때문에 teacher 선택 비중이 그대로 기대 actor 가중치로
남지 않는다. Teacher 좌표를 직접 회귀하는 NLL 힘과 현재의 Q·배정 gradient를 구분한다.

## 6. Marginal SAC와의 정확한 관계

Current actor posterior를

$$
\rho_\theta(i\mid a,s)=\frac{\pi_i(a\mid s)}{H\pi_{\theta,H}(a\mid s)}
$$

라 하면

$$
\boxed{
J_{\rm OT}=J_{\rm SAC}(\pi_{\theta,H})
+T\mathbb E_{\pi_{\theta,H}}
\mathrm{KL}(\rho_\theta(\cdot\mid a,s)\|\Pr(\cdot\mid a,s))+T\log H,
}
$$

$$
J_{\rm SAC}(\pi)=\mathbb E_\pi[T\log\pi-Q]
=T\mathrm{KL}(\pi\|\pi_B)-T\log Z.
$$

따라서 현재 loss는 action marginal과 OT latent 배정을 함께 맞추는 목적이다.
위 KL은 별도의 regularizer를 코드에 추가했다는 뜻이 아니라 현재 식의 정확한 분해다.
Joint 목표에 정확히 적합하고 population source가 균형을 이루면 Boltzmann이 복원되지만,
중간 SGD에서 $J_{\rm OT}$가 감소할 때마다 marginal SAC loss가 감소한다는 보장은 없다.

현재 설계의 정당성은 §4의 조건부 Boltzmann 복원에 두며,
SAC의 marginal-policy 단조 개선 정리를 그대로 적용한다고 주장하지 않는다.
수학적 세부 내용은 [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md)에 있다.

## 7. RL soft TD의 역할과 16-component 밀도 근사

Critic은 최대 entropy 평가와 맞는 soft target을 사용한다.

$$
y=\operatorname{sg}\left[r+\gamma(1-d_{\rm terminal})
\left(\min_l\bar Q_l(s',a')-T\log\hat\pi_\theta(a'\mid s')\right)\right],
\qquad \mathcal L_Q=\sum_l\mathbb E[(Q_l(s,a)-y)^2].
$$

다음 상태마다 fresh normal latent L=16개로 실제 actor Gaussian을 만든다.
첫 component에서 행동을 생성하고 그 component를 포함한 전체 16개 mixture로
$\log\hat\pi$를 계산한다. 생성 component가 bank 안에 포함되는 self-inclusive estimator다.

$$
\log\hat\pi_\theta(a'\mid s')
=\operatorname{LSE}_{l=1}^L\log\mathcal N(u';\mu_l,\operatorname{diag}\sigma_l^2)
-\log L-\log|J_{\tanh}(u')|.
$$

이 밀도는 conditional density 하나가 아니고, 고정4096 bank density도 아니다.
Teacher floor·W·source correction·OT assignment reward는 TD에 넣지 않는다.
Random bank에 조건부로는 정확한 finite-mixture density지만 continuous latent policy의
log-density에는 근사다. Component 교환가능성과 entropy concavity에 의해

$$
\mathbb E[\mathcal H(\pi_L)]\le\mathcal H(\mathbb E[\pi_L])
=\mathcal H(\pi_\theta)
$$

이므로 기대 entropy 하한이다. 개별 표본의 하한 또는 불편 log-density 추정이라는
뜻은 아니다. 유사한 mixture 경계는 [SIVI][sivi]에서 다룬다.

Teacher·actor의 Q 집계는 기본 current twin mean이며 `source_q_eval=min`이면
둘 다 current twin-min이다. TD에는 target twin-min을 사용한다. α 자동 튜닝은 없다.
실제 terminal은 bootstrap하지 않으며 time-limit truncation은 bootstrap한다.

Fixed-Q GMM에는 critic이나 TD가 없다. T=1, $Q(a)=\log p_{\rm GMM40}(40a)$이므로
행동 영역 안의 원래 GMM을 정규화한 Boltzmann 목표를 사용한다. Soft TD 사용 여부로
fixed-Q mode recovery가 저절로 해결되지는 않는다.

## 8. 설정·계산 비용·의사코드

| 설정 | 값 |
|---|---:|
| H / teacher M / 재표집 K / actor 행동 수 | 4096 / 256 / 16 / 16 |
| TD fresh mixture components | 16 |
| OT ε / dual update | .1 / actor당 1회 |
| Actor·critic MLP / Adam | 256×2 GELU / 3e-4 |
| Dual MLP / Adam | 256×2 ReLU→H / 1e-4 |
| Actor 초기 σ / logσ 범위 / teacher floor | .5 / [-5,1] / .05 |
| UTD / train frequency / policy delay | 1 / 1 / 1 |
| Batch / warm-up / replay capacity | 256 / 5,000 / 1,000,000 |
| RL γ / target critic Polyak | .99 / .005 |
| RL T / fixed-Q GMM T | .25 / 1 |

```text
Z ← seed에서 만든 normal 적분점 4096개
actor, twin critic, target critic, persistent dual과 각 Adam 상태 초기화

각 업데이트:
    replay에서 B개 transition 선택                 # GMM은 같은 상태 B lane
    fresh latent 16개 → current actor Gaussian
    첫 Gaussian에서 a_next 표집, 16개 mixture로 logπ_hat 계산
    y ← sg[r + γ(1-terminal)(min targetQ - T logπ_hat)]
    critic Adam, target critic Polyak              # GMM에서는 생략

    f, dual_VJP ← 갱신 전 dual φ(s)
    fresh teacher latent 256개 → 각 Gaussian 행동 하나
    전체 proposal q로 W ← softmax(Q/T - log q)
    W로 16개 importance 재표집, occurrence 질량 1/16
    P_ij ← softmax_i((f_i - ||Z_i-v_tilde_j||²)/ε) / 16
    row_mass_i ← sum_j P_ij
    각 teacher j에서 i_j ~ Categorical(16 P_:j)
    source_weight_j ← sg[(1/H)/row_mass_(i_j)]

    μ_j, σ_j ← actorθ(s,Z_(i_j))                  # 선택된 16개만
    u_j ← μ_j + σ_j ξ_j, fresh noise
    a_j ← tanh(u_j)
    logπ_j ← 선택 Gaussian density + tanh/scale Jacobian
    logPr_j ← logsoftmax_i((f_i-||Z_i-u_j||²)/ε)에서 i_j 선택
    actor_loss ← mean(source_weight_j * [T logπ_j - Q(s,a_j) - T logPr_j])
    actor Adam 1회; Q·Pr의 새 행동 gradient 유지
    dual_grad ← dual_VJP(sg((row_mass - 1/H)/B))
    dual Adam 1회                                  # 같은 갱신 전 f 기준
```

GMM shared f에는 `mean_B(row_mass) − 1/H`를 전달한다.
JAX/Flax/Optax·JIT·`lax.scan`을 유지한다. 4096×16 OT와 새 행동의 assignment query가
필요하지만 전체4096 actor Gaussian의 forward·density·backward는 하지 않는다.
상세 의사코드는 [PSEUDOCODE.md](PSEUDOCODE.md)에 있다.

## 9. 진단·검증·기록의 해석

Teacher W의 density correction과 추가 source correction을 구분한다.
Source 보정계수는 log 행 질량으로 계산하며 clip·self-normalization하지 않는다.
Persistent dual에는 source importance 상한이 없다. Fresh Sinkhorn의 특정
row→column cycle에만 성립하는 상한을 현재 기본 경로에 적용하지 않는다.

Source weight mean/max/ESS, 최소 log 행 질량,
$\log\sum_i(1/H)^2/(\sum_jP_{ij})$, per-lane·batch-mean TV,
actor·dual gradient의 finite 여부를 기록한다. 작은 ε에서 지원 영역이 줄어들면
기대값은 맞아도 희귀한 큰 weight에 의존할 수 있다.

Near-mode 비율과 coverage·mode 질량·MMD²·sliced W₂를 함께 본다.
Teacher 보정 전후·재표집 후·actor 분포를 나누어 검사하고, 필요하면 별도 reference
표본으로 population source 질량과 조건부 목표의 복잡성을 측정한다.
한 batch의 행 balance만으로 실제 Boltzmann 복원을 주장하지 않는다.

기존 [GMM100K_KO.md](GMM100K_KO.md)와 ε 비교는 conditional 목적함수 계열이다.
별도로 시도한 full marginal mixture 변경은 archive로 보존하고 현재 결과와 구분한다.
서로 다른 목적함수의 checkpoint를 같은 실험의 재개로 조용히 취급하지 않는다.
이번 복원 자체는 새 실험 실행을 뜻하지 않는다.

과거 같은 H4096×K16에서 fresh Sinkhorn100은 약19ms/update,
persistent conditional 구현은 약1.8–1.9ms/update, OT-NLL은 약2.04ms였다.
이는 저장된 이전 warm 측정이다. Compile·평가·checkpoint 제외 시간과
batch·H/M/K·solver·objective·GPU 동시 실행 조건을 함께 기록한다.

외부 `gmm40-baseline` 코드는 원본 구현·프레임워크·라이선스를 보존한다.
[복사 manifest](BASELINE_COPY_MANIFEST.json)를 유지하며 복원 때문에 재실행하지 않는다.

## 10. 코드와 관련 문서

| 역할 | 파일 |
|---|---|
| Teacher·source sampling·actor objective | [conditional_sac.py](../../optiq_dime/conditional_sac.py) |
| Persistent dual | [persistent_transport.py](../../optiq_dime/persistent_transport.py) |
| Latent OT | [latent_transport.py](../../optiq_dime/latent_transport.py) |
| RL soft TD·actor 연결 | [algorithm.py](../../optiq_dime/algorithm.py) |
| GMM adapter / runner | [gmm40/v7.py](../../gmm40/v7.py), [train_v7.py](../../gmm40/train_v7.py) |
| 설정 | [final.yaml](../../configs/v7/final.yaml) |
| 표기 / 의사코드 / 수학 감사 | [NOTATION_KO.md](NOTATION_KO.md), [PSEUDOCODE.md](PSEUDOCODE.md), [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md) |
| 검증·학습 기록 | [VALIDATION_KO.md](VALIDATION_KO.md), [GMM_VALIDATION.md](GMM_VALIDATION.md), [GMM100K_KO.md](GMM100K_KO.md) |

[sac]: https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
[sivi]: https://proceedings.mlr.press/v80/yin18b/yin18b.pdf
