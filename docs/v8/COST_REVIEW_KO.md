# v8 OT 비용의 수학적 검토

2026-09-20. 아래의 NLL 검토 이후 사용자는 **원래 z → teacher 거리**를 선택했다.
현재 canonical v8은 C_ij=||z_i-u_j||², 거리 epsilon=.1이며 g(s,z)는 별도로 조건부 SAC
loss로 Gaussian을 학습한다. g의 평균을 source로 쓰는 방식도 선택하지 않았다.
OT 배정과 Gaussian 학습을 분리하는 것이 사용자가 정한 설계 목적이다.
아래 NLL 진단과 fixed-point 논의는 이전 prototype의 기록이며 현재 비용 설명이 아니다.
기존 Gaussian 분해를 유지하는 특수한 성질은 필수 요건에서 제외했지만,
일반적인 balanced assignment의 Boltzmann 복원 충분조건과 Gaussian 표현 오차의
구분은 유지한다. 실제 코드·의사코드는 ALGORITHM_KO.md를 따른다.

## 현재 비용과 실제 진단

현재 source는 raw latent 좌표가 아니라 현재 actor의 실제 Gaussian이다.
pre-tanh teacher 좌표를 u_j라 할 때, alpha=1에서 비용은

    C_ij = -log pi_old,i(b_j)
         = 0.5 sum_d (u_jd-mu_id)^2 / sigma_id^2
           + sum_d log sigma_id + column_constant_j.

고정된 양쪽 marginal 아래에서는 row-only 및 column-only 비용은 optimal P를
바꾸지 않는다. 따라서 현재 배정에서 핵심은 row마다 다른 inverse-variance 거리다.
NLL 비용은 통계적 적합도 비용이지 Euclidean 의미의 이동 거리가 아니다.

Seed0의 update330 직전 고정 actor·Adam·RNG로 재현한 lane26:

- 실제 source sigma(pre-tanh): min .0074119, median .0796418, max .228650.
- Teacher16개는 두 후보의 15회 + 1회 중복. 보정 W 최대 .929674, ESS 1.15675.
- 전체 pair의 Mahalanobis squared distance: median 34.667, max 4618.857.
- Log kernel 범위: -2311.604 ~ -5.833.
- 같은 teacher/actor/RNG에서 500회는 부족했으나 702회에 row relative 1e-3 수렴.

좁은 sigma의 비용 대비와 집중된 finite-sample teacher 질량이 함께 만드는 어려움이다.
이는 cost가 정의되지 않았거나 actor 학습이 발산했다는 증거가 아니다.
과거 v7은 raw z와 teacher u의 squared distance 및 persistent dual을 기본으로 써서
비용과 수렴 검사 모두 현재와 달랐다.

## SAC 연결에서 NLL이 주는 특수한 성질

상태를 생략하고 beta_i=1/H, pi_B(a)=exp(Q(a)/alpha)/Z라 하자.
연속 배정을 r_i(a)=Pr_OT(i|a)로 쓰면 현재 loss는

    L = sum_i beta_i E_{a~pi_theta,i}
        [alpha log pi_theta,i(a) - Q(a) - alpha log r_i(a)].

우리 모델에 직접 KL을 전개하면

    L = alpha KL(beta_i pi_theta,i(a) || pi_B(a) r_i(a))
        + alpha log H - alpha log Z.

또한 actor의 실제 posterior rho_i(a)=beta_i pi_theta,i(a)/pi_theta(a)에 대해

    L = L_SAC(pi_theta)
        + alpha E_{pi_theta} KL(rho(.|a) || r(.|a)) + alpha log H.

즉 임의 OT 배정의 조건부 loss를 marginal SAC와 동일하다고 할 수 없다.

NLL 비용 C=-alpha log pi_old,i, entropic coefficient alpha에서는

    r_i(a) proportional to pi_old,i(a) exp(f_i/alpha).

모집단에서 현재 mixture가 이미 pi_B이면 상수 potential이 source 균형을 만족한다.
이때 r_i는 실제 Gaussian posterior이며 각 조건부 target은 기존 pi_old,i 자체다.
따라서 이미 정확한 mixture의 Gaussian 분해가 fixed point다. 이는 population의
성질이며 256→16 표본으로 푼 경험적 OT에 그대로 적용되는 보장은 아니다.

## 거리 비용도 가능한가

임의의 normalized r_i(a)에 대해

    integral pi_B(a) r_i(a) da = beta_i,
    t_i(a) = pi_B(a) r_i(a) / beta_i,
    pi_theta,i = t_i

이면 sum_i beta_i pi_theta,i = pi_B이다. 이 충분조건 자체는 cost에 의존하지 않는다.
하지만 t_i를 단일 Gaussian으로 표현할 수 있는지는 별개의 문제다.

Pre-tanh 공간에서 Gaussian-to-Dirac squared Wasserstein 비용은 정확히

    W2^2(N(mu_i,Sigma_i), delta_{u_j})
      = ||mu_i-u_j||^2 + tr(Sigma_i).

균형 OT에서는 sum_ij P_ij tr(Sigma_i)=sum_i beta_i tr(Sigma_i)가 상수다.
따라서 이 비용의 P는 mean squared-distance 비용의 P와 같다.
이는 기하학적 배정으로 정당하며 inverse-sigma 확대를 없애지만 Gaussian의
불확실성을 배정에서 구별하지 않는다. Tanh 이후 행동 공간의 W2와 동일한 식은 아니다.

반례: actor가 이미

    pi_B = 0.5 N(0,sigma_1^2) + 0.5 N(0,sigma_2^2), sigma_1 != sigma_2

를 정확히 표현한다고 하자. Mean cost의 두 row는 같으므로 entropic balanced OT는
r_1=r_2=0.5를 주고 각 Gaussian의 목표를 전체 scale mixture로 만든다.
각 target이 단일 Gaussian이 아니므로 원래 정확한 정책을 유지하는 보장이 사라진다.
Gaussian-to-point W2도 row constant가 빠져 같은 반례를 가진다.

따라서 mean/W2 비용 변경은 타당한 ablation이지만 NLL의 오류 수정 또는 기존
SAC 성질을 전부 보존하는 대체라고 주장해서는 안 된다. 거리 비용의 epsilon은
거리 제곱의 단위이며 entropy temperature alpha와 같아야 할 이유도 없다.
Cost만 clipping, normalization 또는 sigma floor로 변경해도 posterior 동등성은
일반적으로 달라진다. Cost와 epsilon을 같은 비율로 바꾸면 P는 그대로이므로
cost/epsilon 대비 문제의 해결이 되지 않는다.

## 당시 검토 판단 (위의 후속 사용자 선택이 우선)

NLL 비용을 수학적으로 틀렸다고 판정할 근거는 없다. 현재 문제의 직접적인 근거는
매우 큰 비용 대비, 집중된 teacher 표본, 상태마다 fresh solve 및 엄격한 균형 검사다.
Boltzmann fixed-point 성질을 우선하면 비용을 보존한 수치 solver 개선을 먼저 검토한다.
기하학적 최소 이동을 우선하면 pre-tanh mean/W2 비용을 별도 비교하되, conditional
target의 Gaussian 표현 오차와 정확한 mixture를 보존하는지까지 같이 검증해야 한다.
이 검토 단계에서는 cost를 교체하지 않았으며, 후속 사용자 선택에 따라 raw-z 거리로 변경했다.

## 근거 문헌

- SAC의 정책 개선 KL: [Haarnoja et al., 2018](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf).
- 비용과 entropy 정규화에 따른 OT: [Cuturi, 2013](https://arxiv.org/abs/1306.0895).
- 작은 유효 entropy에서의 scaling 수치 문제: [Schmitzer, 2019](https://arxiv.org/abs/1610.06519).

Joint-KL 전개, row-constant 제거, 위 mixture 반례와 실측 진단은 현재 OptiQ 설계에
직접 적용한 분석이며, 인용 논문이 OptiQ v8의 수렴을 증명한다는 뜻이 아니다.
