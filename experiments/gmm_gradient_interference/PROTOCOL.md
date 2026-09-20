# Direct GMM: 3-mode gradient 간섭 / latent 분업

## 질문

1. 같은 teacher의 mode별 gradient가 공유 actor parameter에서 충돌하는가?
2. 큰 N에서 서로 다른 latent가 각 mode를 실제로 담당하는가?
3. 한 mode만으로 update하면 다른 mode를 담당하던 latent가 얼마나 움직이는가?

Gradient 수정을 도입하는 실험이 아니다. 실제 training은 기존 0917 Direct GMM 그대로이며,
간섭 측정을 위한 모든 branch는 같은 actor·Adam state에서 출발한 뒤 폐기한다.

## 기존 toy와 연결

`analysis_tools/studies/20260917_nonstationary_q/experiment/{actor,models,problems}.py`와
그 폴더의 v5 dependency를 수정 없이 직접 import한다. 3-mode 공통 prefix를 사용한다.

\[
f(a)=\tfrac13\sum_{c\in\{-0.6,0,0.6\}}\mathcal N(a;c,0.1^2),\quad
Q(a)=0.25\log f(a),\quad a\in[-1,1].
\]

정답은 f를 [-1,1]에서 정규화한 density. 원본 schedule의 중복 Gaussian 6개(d=0)는
동일한 세 mode를 나타내므로 원본 schedule 자체를 호출한다. Q는 20K 동안 고정한다.

| 항목 | 설정 |
|---|---|
| N / M | N64,2048 / M4096 공통 |
| Seeds | 0,1,2,3, 총8개 run |
| 학습 | Random initialization → 20,000 updates |
| Actor | 원본 v5 256×2 GELU, state0, latent1D IID Gaussian, batch1 |
| Sigma | 초기0.5, log sigma [-5,1], learned |
| Proposal | N conditionals의 mixture, IID component sampling, 총M 후보, teacher-only sigma floor0.05 |
| Weight | softmax(Q/0.25−log q_F), beta1, tanh Jacobian 포함 |
| Optimizer | 원본 Adam3e-4, clipping/annealing/EMA/OT/gradient modification 없음 |
| Precision | 원본 JAX float32/default matmul, GMM40의 highest override를 상속하지 않음 |
| 평가 | 32768 새 latent/noise action histogram,256bins, KDE smoothing 없음 |
| Fixed probe | 동일한 seed별2048 latent, training RNG와 별도 |
| 실행 | 31.148.50.247:11717 GPU3, CPU6–9, 한 run씩5K segment round robin |

M을 고정한 것은 N의 효과와 teacher 후보 수 효과를 분리하기 위함이다.
기존 크기 표 전체를 재현하는 실험은 아니다. N은 새로운 latent 표본 수이지
독립적으로 학습되는 network/component head의 개수가 아니다.

## Gradient 분해

Mode basin 경계는 [-1,−0.3,0.3,1]. 모든 후보를 한 영역에 배정한다.

\[
L_m=-\sum_{j:b_j\in\mathcal B_m}w_j\log q_{mix,\theta}(b_j),
\qquad L=\sum_mL_m,\quad g_m=\nabla_\theta L_m.
\]

각 L_m도 N개 전체 Gaussian의 mixture density를 쓴다. Teacher w를 mode 질량으로
재정규화하지 않는다. Teacher 후보/weight/latent/label을 detach한다. Teacher의
tanh Jacobian은 parameter-independent이므로 gradient는 원본 NLL과 같다.

Full parameter, shared trunk, mu head, sigma head별로 3×3 Gram/cosine, norm을 저장한다.
음의 cosine만으로 해로운 간섭이라고 판정하지 않고 실제 Adam 결과를 함께 본다.

## 한 update의 영향

각 진단 시점(step0,1,10,100,500,1K,2K,5K,10K,15K,20K)에 다음 실제 update가 쓸
teacher와 latent를 미리 계산한다. Training RNG는 소비하지 않는다.

1. 원본 full Direct GMM Adam 한 번.
2. 새 gradient=0인 Adam 한 번: 이전 momentum만으로 인한 이동 대조군.
3. 각 mode loss만 사용한 Adam 한 번: 세 branch.
4. 각 mode의 plain SGD 방향, parameter displacement norm은 full Adam과 동일: 세 branch.

총8개 branch 각각에 대해 같은 teacher/latent의 mode별 NLL 변화, g_m·delta_theta의
1차 예측, 별도 정답 reference NLL 변화를 저장한다. 정답 reference는1024 bin의
정확한 CDF 질량과 midpoint, 고정2048 evaluation latent를 사용한 보조 quadrature이며
정답 density의 정확한 likelihood 적분이라고 주장하지 않는다.

단독 mode Adam에는 이전 full-gradient Adam history가 그대로 있다. Zero-gradient
control은 그 영향을 보여주지만, Adam의 비선형 preconditioning 때문에 단순 차감으로
인과적인 순수 gradient 효과가 분리된다고 주장하지 않는다. Norm-matched SGD는
그와 다른 방향성 대조이며 실제 training 변경이 아니다.

## Latent 담당 mode와 이동

고정 latent별 conditional Gaussian의 basin 확률을 Gaussian CDF로 계산한다.
한 basin 확률 >=0.8이면 해당 mode specialist, 아니면 broad/ambiguous로 표시한다.
단순히 가장 가까운 mean으로 넓은 Gaussian까지 강제 분류하지 않는다.

각 branch에서 grouping은 update 전 기준으로 고정하고 tanh(mu)의 RMS 이동,
log sigma 변화, 원래 담당 basin 확률 변화를 기록한다. 빈 specialist group은
측정 불가(NaN/그림 빈칸)로 표시하며0효과로 해석하지 않는다. Tanh(mu)는 action
평균이 아니라 conditional 대표 위치다. 같은 latent identity를 시간축에서 추적한다.

현재 training N개 latent에 대해서도 responsibility의 mode별 합,
alpha_i=sum_j w_j gamma_ij, underused fraction(alpha<0.1/N), usage ESS를 기록한다.
이는 고정 evaluation latent와 별도이고 row 번호를 시간축 identity로 사용하지 않는다.

## 검증과 자원

실행 전 두 N에서 mode gradient 합과 원본 gradient 일치, 유한차분,
원본3step와 block3step 일치, teacher detach, assignment mass conservation,
checkpoint 다음 update 일치, 진단 후 actor/Adam/RNG 불변을 검사한다.
그림의 density는 모두 실제32768 actions의 histogram이며, CDF는 reference와
component 역할 진단에만 사용한다.

GPU3에서 사전검증 →8개 학습을 실행한다. GPU0–2 MuJoCo와 큐는 변경하지 않는다.
매500step checkpoint,5K segment round robin. Training 시간과 diagnostic 시간을 분리한다.
각 segment 뒤 MD와 그림이 embedded된 HTML 보고서를 갱신한다. 통계적 결론은
4seeds의 같은 step을 비교하며, seed0 figure 하나만으로 일반화하지 않는다.

소스·설정·launch·본 protocol을 heejoon에 commit/push 후 실행한다. SHA,manifest,
queue PID/run ID를 기록한다. 데이터는 legacy_monge/gradient_interference/SHA/ 아래에
저장하고 기존 dildata collector로 회수한다. SSH/W&B keys는 포함하지 않는다.
Toy는 로컬 상세 로그/리포트에 기록하며 기존 MuJoCo W&B run과 섞지 않는다.
