# OptiQ v8: 상태별 Gaussian OT와 조건부 SAC

사용자 요청에 따른 v8 설계. v7 자료와 실행 기록은 별도로 보존한다.
GMM40에서 먼저 검증하고, RL 학습 캠페인은 별도 단계다.

## 정책과 보정 대상

공유 MLP는 `g(s,z)=(mu,log_sigma)`를 출력한다. 실제 행동은 tanh-Gaussian이다.
고정 normal 적분점 `z_i`, H=4096은 prior 적분을 근사하며 각 질량은 1/H다.
Gaussian 출력은 현재 actor와 현재 상태에서 다시 계산한다.

\[
\pi_{\theta,H}(a|s)=H^{-1}\sum_i\pi_\theta(a|s,z_i),\qquad
\pi_B(a|s)\propto\exp(Q(s,a)/\alpha).
\]

Fresh latent 256개에서 Gaussian을 만들고 각각 행동 하나를 표집한다.
실제 proposal 전체 혼합밀도 q를 사용해

\[
W_j=\operatorname{softmax}_j[Q(s,b_j)/\alpha-\log q(b_j|s)]
\]

로 보정한 뒤 K=16개 occurrence를 재표집한다. 중복을 유지하고 열 질량은 1/K다.
W는 이 빈도에 한 번 반영한다. Teacher proposal sigma floor .05는 q와 sampling에만
들어가며 source Gaussian, actor entropy, TD density에 넣지 않는다.
GMM은 물리 좌표 x=40*tanh(u), Q=log p_GMM40(x), alpha=1을 사용한다.
행동 density는 tanh Jacobian과 물리 scale Jacobian을 포함한다.

## 상태별 fresh OT

현재 actor를 고정한 실제 Gaussian으로 비용을 만든다.

\[
C_{ij}=-\alpha\log\pi_{{\rm old},i}(\tilde b_j|s),\quad
P^*=\arg\min_{P\mathbf1=\beta,\,P^\top\mathbf1=\nu}
\langle P,C\rangle+\alpha\mathrm{KL}(P\|\beta\otimes\nu),
\quad \beta_i=1/H,\;\nu_j=1/K.
\]

따라서 Gaussian NLL 단위의 dimensionless epsilon은 1이다. 비용과 entropic 계수를
함께 alpha로 스케일하므로 구현에서는 Gaussian log likelihood를 kernel로 사용한다.
이 설정은 과거 squared-distance epsilon=1 실험과 다르다.
NLL은 배정 비용이며, actor loss는 teacher NLL 회귀가 아니다.
통계적 적합도 비용을 Euclidean/Wasserstein 이동 거리와 동일시하지 않는다.

Replay batch의 상태마다 독립적으로 log-domain Sinkhorn을 새로 푼다. Dual은
그 solve의 내부 변수이며 별도 네트워크·optimizer·장기 저장 상태가 없다.
기본 min/max iterations=10/500, 상대 marginal 허용오차=1e-3다. 행과 열을 검사한다.
미수렴 또는 비유한 업데이트는 actor/Adam/RNG를 보존한 채 거부하고 실행기가 중단한다.
초기화가 쉬운 상태의 성공만으로 학습 후 모든 상태의 수렴을 가정하지 않는다.

일반 행동에 대한 연속 배정은

\[
\Pr_{\rm OT}(i|a,s)=\operatorname{softmax}_i
[\log\pi_{{\rm old},i}(a|s)+f_i(s)/\alpha].
\]

코드가 반환하는 `source_potential`은 dimensionless f/alpha다.
새 행동에서도 teacher 비용과 동일한 actual Gaussian/Jacobian 규약을 사용한다.

## Actor와 gradient

각 teacher column에서 I_j ~ Categorical(K P_:j)를 하나씩 뽑는다.
균형 조건 아래 평균 source 선택 확률은 1/H이므로 추가 source importance ratio가 없다.
선택한 latent에서 새 Gaussian noise로 행동 a_j를 생성하고

\[
L_\theta=\frac1K\sum_j[\alpha\log\pi_{\theta,I_j}(a_j|s)
-Q(s,a_j)-\alpha\log\Pr_{\rm OT}(I_j|a_j,s)]
\]

를 미분한다. Current twin critics의 minimum을 teacher와 actor에 동일하게 사용한다.
Q 파라미터, 이전 actor Gaussian 파라미터, OT 해, teacher 표집은 stop-gradient다.
새 행동을 통한 Q와 assignment의 gradient는 반드시 유지한다.
현재 선택된 Gaussian의 평균과 sigma 모두 같은 loss로 학습한다.

H개 source Gaussian의 frozen forward는 필요하다. 파라미터 backward는 선택된 K개다.
GMM의 모든 lane이 동일한 상태임을 명시한 adapter에서는 source forward만 공유할 수
있다. Teacher와 OT는 lane마다 독립적으로 유지하며, 일반 RL 상태에는 공유하지 않는다.

## Soft TD와 보장의 범위

\[
y=r+\gamma(1-d)[\min Q_{\rm target}(s',a')-\alpha\log\pi_\theta(a'|s')].
\]

RL은 기존 fresh 16-component self-inclusive mixture density 추정으로 이 soft TD를
근사한다. 조건부 하나의 entropy와 전체 mixture entropy를 혼동하지 않는다.
GMM은 fixed-Q이므로 critic이 없다.

모집단 질량 barP_i=integral pi_B(a)Pr_OT(i|a)da와 조건부 목표
t_i=pi_B Pr_OT / barP_i를 정의하면, barP_i=1/H이고 pi_i=t_i일 때 전체 mixture가
Boltzmann을 복원한다. 비용 epsilon=1에서 기존 mixture가 정확히 pi_B이면 상수 dual이
균형을 만족하고 각 t_i가 기존 Gaussian과 같아지는 이상적 fixed point가 존재한다.
유한 teacher 16개에서는 이 정리가 그대로 성립하지 않으며 noisy dual로 drift할 수 있다.

실제 actor posterior rho_i=pi_i/(H*pi_H)에 대해

\[
L_\theta=L_{\rm SAC}(\pi_H)
+\alpha E_{\pi_H}\mathrm{KL}(\rho\|\Pr_{\rm OT})+\alpha\log H.
\]

따라서 조건부 SAC 확장이며 marginal SAC와 동일하다는 주장이나 원래 SAC 정책개선
정리의 직접 적용을 하지 않는다. OT 단계와 actor 단계를 하나의 같은 KL에 대한
block-coordinate descent라고 부르지도 않는다. Teacher 미발견, 조건부 다중모드,
공유 MLP 간섭, finite-sample balance와 soft-TD density 근사 오차가 남는다.

## 한 업데이트 의사코드

```python
old_mu, old_log_sigma = stopgrad(actor(s, Z_4096))
b, u, log_q = fresh_one_per_latent_proposal(actor, s, M=256)
W = stopgrad(softmax(Q(s, b) / alpha - log_q))
u_tilde = importance_resample(u, W, K=16)
log_kernel = log_actual_source_gaussians(u_tilde, old_mu, old_log_sigma)
P, f_over_alpha, residuals = fresh_balanced_sinkhorn(log_kernel)
require_converged_and_finite(residuals)
I = sample_one_source_per_column(P)
a, log_pi_conditional = reparameterized_actor(s, Z_4096[I])
log_assignment = log_softmax(log_actual_source_gaussians(a, old_mu, old_log_sigma)
                             + stopgrad(f_over_alpha), axis='source')
loss = mean(alpha * log_pi_conditional - Q(s, a)
            - alpha * gather_source(log_assignment, I))
update_actor_once(loss)  # no teacher NLL and no source ratio
```

MLP 256x2, Adam 3e-4, 초기 sigma=.5, log sigma bounds[-5,1]은 기존 비교 설정을 유지한다.
