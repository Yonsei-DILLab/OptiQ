# v5 구현 세부 사항

이 문서는 [의사코드](PSEUDOCODE.md)의 수학과 실제 코드 경로를 설명한다.
현재 기본값은 T=.25, Sinkhorn epsilon=.1, 100회, gradient clip 없음이다.

## 1. 연속 mixture와 tanh

한 상태 s에서 z~Normal(0,I)를 샘플링하고 신경망이 mu(s,z), log_sigma(s,z)를
출력한다. z가 고정되면 u|s,z는 대각 Gaussian이고, z를 주변화한 분포는
연속 mixture다. 서로 다른 z가 서로 다른 평균을 낼 수 있지만, 학습 결과가
반드시 여러 mode를 가진다는 보장은 없다. M=16은 매 업데이트에서 뽑는
latent 개수이며 영구적인 16-component codebook이 아니다.

```text
z ~ Normal(0,I)
u = mu(s,z) + sigma(s,z) * epsilon, epsilon ~ Normal(0,I)
a = tanh(u)
```

Gaussian은 범위가 무한하므로 tanh로 각 좌표를 (-1,1)에 보낸다. 환경에는
action bounds로 unscale한 값을 전달하고 replay에는 정규화된 행동을 저장한다.
Tanh는 각각의 행동 샘플에 적용하며 component 평균들을 먼저 합치지 않는다.
Hard clip과 달리 실수의 서로 다른 값을 수학적으로 같은 끝점에 모으지 않고,
변수변환으로 밀도 계산도 가능하다. 실제 부동소수점에서는 포화가 발생할 수 있다.

다만 tanh는 거리·분포 모양·밀도 mode 개수를 보존하지 않는다. mu=2와 3은
약 .964와 .995가 되어 행동 공간의 차이가 작아진다. 또한
`tanh(mu(s,z)) != E_epsilon[tanh(mu(s,z)+sigma(s,z)*epsilon)]`가 일반적이다.
따라서 mean-action이라는 이름은 squashed location을 뜻하며, epsilon까지
적분한 정확한 조건부 평균이나 mixture 전체의 평균·MAP를 뜻하지 않는다.

코드: [policy.py](../../optiq_dime/policy.py)의 `SemiImplicitActor`, `sample_action`.

## 2. Proposal 밀도와 beta=1 가중치

Replay의 각 상태에 대해 z_i를 M=16개 뽑는다. Teacher 전용 표준편차는
`sigma_proposal_i = max(sigma_i, .05)`다. 후보 K=64개는 각자 component를
균등 독립 선택한 뒤 해당 Gaussian에서 u_j를 샘플링하고 b_j=tanh(u_j)로 만든다.
따라서 한 component에서 항상 4개씩 뽑는 stratified 방식이 아니다.
Anchor는 넣지 않는다. `exact`는 이 유한 proposal의 샘플링·밀도 의미다.

고정된 M개 component에 대한 행동 공간 밀도는

\[
q_A(\tanh u\mid s)=
\frac{\frac1M\sum_i\mathcal N(u;\mu_i,\operatorname{diag}\sigma_{p,i}^2)}
     {\prod_d(1-\tanh^2 u_d)}.
\]

차원별 Gaussian log-density를 먼저 합한 후 component 축에 logsumexp를
적용한다. 좌표마다 component를 따로 섞지 않는다. 샘플링과 log q 양쪽이
같은 teacher std floor를 사용한다. 이 유한 q는 정확히 계산하지만 원래
연속-z actor의 주변밀도를 정확히 계산한 것은 아니다.

Teacher reference는 정규화된 action 공간의 uniform 측도에 대한
`p_ref(a|s) ∝ exp(Q_mean(s,a)/T)`다. Q_mean은 현재 twin critic의 산술평균이다.
q에서 뽑은 후보로 이 reference를 적합하려고

\[
w_j=\operatorname{softmax}_j\left(Q_{mean}(s,b_j)/T-\log q_A(b_j\mid s)\right)
\]

를 쓴다. Beta=1은 밀도 보정 전체를 쓴다는 의미다. Prioritized replay의
importance weights가 아니며, ESS에 따라 beta를 자동 조절하지 않는다.
유한 K의 self-normalized importance sampling이므로 exact reference fit이나
정책 개선 보장은 아니다. 상태별 공통 Q offset은 softmax에서 상쇄된다.

원래 u_j를 보관하여 `atanh(tanh(u_j))` 복원을 피한다. Log-Jacobian은
`sum_d 2*(log(2)-u_d-softplus(-2*u_d))`로 계산한다.
`proposal_clip=.5`라는 상속 필드는 이 conditional Gaussian 경로에서 teacher
샘플을 잘라내지 않는다. Teacher floor .05도 actor sigma의 하한과는 다르다.

코드: [semi_implicit.py](../../optiq_dime/semi_implicit.py)의
`ConditionalGaussianProposal`, `conditional_mixture_log_prob`.

## 3. Mean OT의 배정 대상

각 상태마다 학생 x_i=tanh(mu_i), teacher b_j=tanh(u_j)의 제곱거리 합
`C_ij=sum_d (x_id-b_jd)^2`를 사용한다. 비용을 차원 수나 평균 비용으로 나누지 않는다.
행 marginal은 1/M, 열 marginal은 w다. Log-domain Sinkhorn을 정확히 100회
반복하며 residual에 따른 조기 종료나 iteration 자동 증가는 없다.
작은 epsilon은 비용 차이에 더 민감한 배정을 유도하지만, 유한 반복에서
marginal이 충분히 맞는지는 기록된 residual로 확인해야 한다.

```text
P = Sinkhorn(C, row_mass=1/M, column_mass=w, epsilon=.1, iterations=100)
R_ij = P_ij / max(sum_j P_ij, 1e-20)
```

코드는 열 weights를 [1e-20,1]로 제한한 뒤 정규화하고, 최종 실제 행 합으로
R을 정규화한다. 아주 작은 weights와 유한 반복 때문에 이상적인 marginal과
차이가 날 수 있다. Sigma는 **학생 위치**에 직접 들어가지 않지만 teacher
후보·밀도·weights에는 들어간다. 같은 mu에 서로 다른 sigma를 가진 학생은
동일한 비용 행을 가지므로 이 비용만으로는 구분되지 않는다.

기존 student Gaussian epsilon draw는 유지한다. 이 값은 sampled-action 진단에
사용되며 mean OT로 바꿨다는 이유로 RNG 순서를 앞당기지 않는다.
`transport_target_mode=argmax`도 남아 있지만 full-row NLL의 목표를 argmax
한 점으로 바꾸지는 않는다. 선택 행동·hard projection 관련 진단에만 쓰인다.

코드: [transport.py](../../optiq_dime/transport.py),
[algorithm.py](../../optiq_dime/algorithm.py)의 `update_actor`.

## 4. NLL이 mu와 sigma를 학습하는 방법

고정된 teacher와 R에 대해 `m_i=sum_j R_ij*u_j`,
`V_i=max(sum_j R_ij*u_j^2-m_i^2,0)`를 좌표별로 만든다.

\[
L_{actor}=\frac1{BM}\sum_{s,i,d}
\left[\frac{(\mu_{id}-m_{id})^2+V_{id}}{2\sigma_{id}^2}
      +\log\sigma_{id}+\tfrac12\log(2\pi)\right].
\]

행의 모든 teacher 후보를 사용한다. Teacher component의 sigma를 복사하는
방식이 아니라 배정된 샘플의 잔차 분산에 맞춘다. 자유로운 출력과 양의 V에서
최적해는 mu=m, sigma²=V이며, 실제 네트워크는 출력 제약·공유 파라미터·유한
Adam step 때문에 이 값을 정확히 달성하지 않을 수 있다.

고정된 m,V에 대한 좌표별 미분은

\[
\frac{\partial L}{\partial\mu}=\frac{\mu-m}{\sigma^2},\qquad
\frac{\partial L}{\partial\log\sigma}
=1-\frac{(\mu-m)^2+V}{\sigma^2}.
\]

Teacher 생성, critic, 중요도 weights, OT plan, m,V로 향하는 gradient는
차단한다. Actor의 현재 mu·log_sigma와 공유 hidden layers에만 gradient가
흐른다. Q-gradient를 직접 더하거나 Sinkhorn을 역전파하지 않는다.

Action 공간의 conditional NLL에는 tanh Jacobian 항이 있으나 고정 teacher u의
함수이므로 학생 파라미터에 대해 상수다. 따라서 NLL에서는 생략해도 같은
gradient다. Importance weight의 q_A에서는 Jacobian을 빼면 다른 reference를
학습하게 되므로 생략하지 않는다. NLL 전체가 학생 tanh를 거쳐 미분되는 것은
아니어서 tanh 포화만으로 모든 actor gradient가 사라진다고 설명할 수 없다.

이는 OT로 배정한 **조건부 likelihood 적합**이며 일반적인 marginal forward KL이나
SAC reverse KL과 동일한 목적은 아니다. Diagonal Gaussian은 조건부 좌표 공분산을
직접 표현하지 못하지만, z에 따른 mixture가 주변분포의 상관관계를 만들 수 있다.

코드: [distillation.py](../../optiq_dime/distillation.py)의 `conditional_ot_nll`.

## 5. Sigma가 사용되는 위치

| 경로 | z | Gaussian epsilon | sigma 사용 |
|---|---|---|---|
| Warmup 이후 환경 수집 | 샘플링 | 샘플링 | actor sigma |
| TD next action | 샘플링 | 샘플링 | 현재 actor sigma |
| Teacher 후보 / q_A | M개 샘플링 | 후보별 샘플링 | max(actor sigma,.05) |
| OT student 위치 | M개 샘플링 | 위치에서 제외 | 직접 사용하지 않음 |
| NLL | OT에 사용한 동일 z | 고정 teacher u에 반영 | mu와 sigma 모두 적합 |
| zero-z 평가 | 0 | 0 | 행동에 사용하지 않음 |
| sampled-z 평가 | 행동마다 샘플링 | 0 | 행동에 사용하지 않음 |

초기 sigma는 .5이며 log-std head weight=0, bias=log(.5)다. 출력 log-std는
[-5,1] 범위다. No-clip은 **gradient** 제한을 없앤 것으로, 이 출력 범위,
tanh 변환, teacher std floor는 그대로 유지한다.

## 6. TD, optimizer와 업데이트 순서

현재 actor에서 next action 한 개를 샘플링하고 target twin-min으로
`y=r+.99*(1-terminal)*min(Q_target1,Q_target2)`를 만든다. True terminal만
bootstrap을 끄고 time-limit truncation은 마지막 실제 관측으로 bootstrap한다.
Q는 scalar이며 설정에 남은 v_min/v_max로 값을 제한하지 않는다.

Critic loss는 두 네트워크의 batch MSE를 합한다. Critic 업데이트 후 target
critic을 tau=.005로 갱신하고, 갱신된 live critic의 **mean**을 teacher에 쓴다.
Actor와 critic은 같은 replay batch에서 각각 한 번 업데이트한다. Policy delay=1,
UTD=1, warmup=actor start=5K이므로 1M에서 각각 995K updates다.

현재 actor·critic은 LR=3e-4, beta1=.9, beta2=.999, epsilon=1e-8인 Adam이다.
`ac_grad_norm=null`이면 optimizer factory는 clipping transform 없이 Adam을
반환한다. Critic의 twin-min 선택과 gradient clip은 서로 다른 연산이다.
과거 clip=2에서는 두 critic의 gradient tree를 합쳐 하나의 global norm을 썼다.

TD의 entropy bonus, actor의 직접 entropy bonus, soft guard의 g>0 검사,
업데이트 거절·optimizer rollback은 없다. T에 따른 reference weights,
entropic OT와 NLL의 log_sigma 항은 계속 존재한다. 이들을 TD entropy bonus와
동일시하지 않는다. Target actor 저장은 checkpoint 호환용이며 현재 v5의 TD
next action은 live actor를 사용한다.

코드: [optimizers.py](../../optiq_dime/optimizers.py),
[algorithm.py](../../optiq_dime/algorithm.py)의 `_train`, `update_critic`.

## 7. 평가와 로그 해석

평가는 step 1, 이후 매 5K마다 두 모드 각각 10 episode다. 두 모드의 환경
reset seed와 policy seed를 맞추고 학습/수집 RNG에서 분리한다. Sampled-z 모드는
episode당 한 번이 아니라 행동마다 z를 새로 뽑는다. 두 평가 모두 epsilon=0이다.
정기 평가는 해당 step의 learner update 전, checkpoint는 update 후에 저장한다.

`eval/mean_reward`는 zero-z의 호환 alias이며 두 모드 중 높은 점수를 고르는
값이 아니다. `rollout/ep_rew_mean`은 Gaussian 정책으로 실제 수집한 episode
보상이다. 따라서 rollout과 epsilon=0 평가 점수가 다를 수 있다.

| 로그 (`train/` prefix) | 읽는 방법 |
|---|---|
| `temperature` | 실제 teacher T; Sinkhorn epsilon은 config에서 확인 |
| `source_ess_absolute`, `max_source_weight` | K=64 후보 중요도 가중치 집중도 |
| `q_only_ess_fraction`, `density_only_ess_fraction` | Q/T와 -log q 영향의 개별 진단 |
| `actor_std_mean/min/max` | replay batch·z에서의 pre-tanh sigma; 행동 공간 std가 아님 |
| `actor_latent_mean_variance_fraction` | pre-tanh between-mu variance / (between + within); mode 수가 아님 |
| `policy_spread_l2` | Gaussian epsilon을 포함한 sampled action spread; mean-only 다양성이 아님 |
| `student_action_saturation_fraction` | sampled student action에서 abs(a)>.99 비율; OT mean 위치 자체의 포화율이 아님 |
| `ot_cost_mean` | mean 위치와 teacher 행동의 raw squared distance |
| `ot_row_marginal_error`, `ot_col_marginal_error` | 100회 이후 plan의 marginal 오차 |
| `hard_projection_mass_tv` | hypothetical row-argmax projection 진단; 실제 NLL은 full row 사용 |
| `actor_loss`, `critic_loss`, `twin_q_abs_diff` | 상태 분포와 scale에 의존; 환경 보상 개선의 직접 증거가 아님 |

ESS가 낮다는 이유만으로 성능 실패를 단정하거나, loss가 낮다는 이유만으로
더 좋은 정책이라고 결론 내리지 않는다. 실제 평가와 대조해야 한다. 과거 모델의
분석 근거와 남은 불확실성은 [수치 진단](RESIDUAL_DIAGNOSIS_KO.md)에 정리했다.
