# OptiQ v8: raw z → teacher 거리 OT와 조건부 SAC

사용자가 선택한 역할 분리: **OT는 latent prior의 좌표에서 teacher 목표로 배정하고,
g(s,z)는 배정된 조건부 목표를 따라 Gaussian을 학습한다.**
OT 비용에 현재 Gaussian의 mean, sigma 또는 likelihood를 넣지 않는다.
기존 Gaussian-NLL-cost prototype은 별도 commit·실험 기록으로 보존한다.

## 정책과 importance correction

일반 MLP가 g(s,z)=(mu,log_sigma)를 출력하고 z~N(0,I),
u=mu+sigma*noise, a=tanh(u)로 행동을 만든다.
고정된 normal 적분점 z_i, H=4096이 latent prior를 근사하며 각 source 질량은 1/H다.
행동 수집·평가에서는 기존 continuous normal prior를 유지한다.

\[
\pi_\theta(a|s)=E_{z\sim N(0,I)}\pi_\theta(a|s,z),\qquad
\pi_B(a|s)=Z(s)^{-1}\exp(Q(s,a)/\alpha).
\]

Teacher용 fresh latent 256개에서 Gaussian을 만들고 각 Gaussian에서 행동 하나씩
표집한다. 실제 proposal 전체 혼합밀도 q를 평가한 뒤

\[
W_j=\operatorname{softmax}_j[Q(s,b_j)/\alpha-\log q(b_j|s)]
\]

로 보정하고 K=16개 occurrence를 importance 재표집한다. 중복은 그대로 유지한다.
보정 W는 이 재표집 빈도에 한 번 반영되며 OT 열 질량은 1/K다.
Teacher proposal sigma floor .05는 sampling과 q에만 적용한다.
Tanh Jacobian과 물리 action-scale Jacobian은 q와 actor density에서 유지한다.
GMM은 x=40*tanh(u), Q=log p_GMM40(x), alpha=1을 사용한다.

## 상태별 fresh raw-latent OT

재표집된 teacher의 pre-tanh 좌표를 tilde u_j라 하면

\[
C_{ij}=\|z_i-\tilde u_j\|^2,\qquad
P^*=\arg\min_{P\mathbf1=\beta,\,P^\top\mathbf1=\nu}
\langle P,C\rangle+\varepsilon\,\mathrm{KL}(P\|\beta\otimes\nu),
\quad\beta_i=1/H,\quad\nu_j=1/K.
\]

거리 epsilon=.1과 Boltzmann/entropy temperature alpha는 서로 다른 파라미터다.
Cost는 pre-tanh 좌표의 제곱거리이며 batch 평균으로 다시 정규화하지 않는다.
물리 좌표 x의 40배를 OT 비용에 곱하지 않는다.

각 replay 상태마다 log-domain Sinkhorn을 처음부터 독립적으로 푼다.
Dual은 해당 solve의 내부 변수일 뿐 상태 사이·업데이트 사이에 저장하지 않는다.
Min/max iterations=10/2000, 행·열 상대 marginal 오차 한계=1e-3다.
내부 종료 판정과 반환 plan 검사는 같은 normalized-plan 계산을 사용한다.
미수렴 또는 비유한 입력/업데이트는 actor·Adam·RNG를 보존하고 거부한다.

새로운 행동 a에 대한 연속 배정도 동일한 비용을 사용한다.

\[
r_i(a|s)=\Pr_{\rm OT}(i|a,s)
=\operatorname{softmax}_i
\left[\frac{f_i(s)-\|z_i-\operatorname{atanh}(a)\|^2}{\varepsilon}\right].
\]

실제 코드에서는 새 행동을 만들 때의 pre-tanh u를 직접 사용해 atanh를 다시 하지 않는다.
Solver의 source_potential은 dimensionless f/epsilon이며 query에서도 이 규약을 맞춘다.
Teacher용 P와 actor query에 서로 다른 거리나 Gaussian likelihood를 사용하면 안 된다.

## Gaussian actor 학습

각 teacher column에서 I_j~Categorical(K P_:j)를 하나씩 뽑는다.
균형 OT이면 column 평균 source 선택 확률이 1/H라 추가 source importance ratio가 없다.
선택한 16개 z에서 **새 Gaussian noise**로 행동 a_j를 만들고

\[
L_\theta=\frac1K\sum_j
\left[\alpha\log\pi_\theta(a_j|s,z_{I_j})
-Q(s,a_j)-\alpha\log r_{I_j}(a_j|s)\right]
\]

를 미분한다. Critic 파라미터, teacher 표집, OT 해, source z는 stop-gradient지만
새 행동을 통한 Q와 log r의 gradient는 유지한다. Mean과 sigma 모두 이 loss로 학습한다.
OT 비용에서 sigma가 빠져도 actor의 Gaussian density·entropy에서 sigma가 사라지는 것은 아니다.
Teacher와 actor는 같은 current twin critics의 minimum을 사용한다.

H=4096개 source의 actor forward는 필요하지 않다. Gaussian forward는 teacher 후보
256개와 선택된 actor latent 16개에서만 수행한다. 4096×16 OT 계산은 여전히 필요하다.
GMM batch lane마다 teacher/OT는 독립적이다. RL에서도 매 상태마다 새로 배정한다.

## Soft TD와 Boltzmann 복원의 정확한 범위

\[
y=r+\gamma(1-d)
[\min Q_{\rm target}(s',a')-\alpha\log\pi_\theta(a'|s')].
\]

RL은 기존 fresh 16-component self-inclusive mixture density 추정으로 soft TD를
근사한다. GMM은 fixed-Q이므로 critic 업데이트가 없다.

모집단에서 normalized 배정 r_i가

\[
\int\pi_B(a|s)r_i(a|s)\,da=\beta_i,\qquad
t_i(a|s)=\pi_B(a|s)r_i(a|s)/\beta_i
\]

를 만족하고 각 actor conditional이 t_i에 맞으면

\[
\sum_i\beta_i\pi_\theta(a|s,z_i)
=\sum_i\pi_B(a|s)r_i(a|s)=\pi_B(a|s).
\]

이 충분조건은 Gaussian NLL 비용을 요구하지 않는다. 또한 현재 조건부 loss는

\[
L_\theta=\alpha\,\mathrm{KL}
\big(\beta_i\pi_\theta(a|s,z_i)\,\|\,\pi_B(a|s)r_i(a|s)\big)
+\alpha\log H-\alpha\log Z(s)
\]

로 해석할 수 있다. 다만 실제 Gaussian이 t_i를 표현할 수 있는지, 유한 teacher16개의
균형이 모집단 균형을 근사하는지, 신경망 SGD가 해를 찾는지는 별도 조건이다.
거리 비용은 이미 정확한 mixture의 임의 Gaussian 분해를 그대로 보존하지 않을 수 있다.
Marginal SAC와 loss가 정확히 같다고 주장하거나 SAC 정책개선 정리를 그대로 적용하지 않는다.
고정 geometry 배정과 Gaussian 학습의 역할 분리가 선택한 설계 원칙이다.

## 한 업데이트 의사코드

```python
z = fixed_normal_integration_sites(H=4096)  # no g(s, z) over H
b, teacher_u, log_q = fresh_one_per_latent_proposal(g, s, M=256)
W = stopgrad(softmax(Q(s, b) / alpha - log_q))
u_tilde = importance_resample(teacher_u, W, K=16)
C = squared_distance(z, u_tilde)
P, f_over_epsilon, residuals = fresh_balanced_sinkhorn(-C / epsilon)
require_converged_and_finite(residuals)
I = sample_one_source_per_column(P)
u, a, log_pi_conditional = reparameterized_actor(g, s, z[I])
log_assignment = log_softmax(
    stopgrad(f_over_epsilon) - squared_distance(z, u) / epsilon,
    axis="source",
)
loss = mean(alpha * log_pi_conditional - Q(s, a)
            - alpha * gather_source(log_assignment, I))
update_actor_once(loss)  # no teacher NLL; no extra source ratio
```

MLP256×2, Adam3e-4, 초기 sigma=.5, log sigma bounds[-5,1]을 유지한다.
GMM alpha=1, 거리 epsilon=.1, H4096/M256/K16을 기본으로 한다.
