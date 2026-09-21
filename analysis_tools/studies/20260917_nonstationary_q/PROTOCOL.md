# 1D non-stationary Q: OT와 Direct GMM의 분포 추적

사용자가 승인한 2026-09-17 설계의 실행 명세다. 결론과 가설을 구분한다. 기존 연구·Vast MuJoCo 실험은 수정하지 않는다.

## 질문과 등록 범위

Q가 변할 때 marginal mixture fitting과 balanced OT assignment 중 어느 쪽이 mode의 유지·획득·회복을 돕는가? 큰 sigma로 빈 공간까지 덮는 것과 실제 여러 peak의 복구를 구분한다.

- 방법 7개: Direct GMM learned sigma; Exact/Sinkhorn × learned sigma/fixed 0.5/fixed 0.1.
- 크기 4개: N×M = 16×64, 256×16384, 1024×4096, 2048×2048.
- Seeds 0,1,2,3. 크기나 방법에 따라 seed를 선별하지 않는다.
- 최종 비교 궤적: mass / split / learned-Q replay / closed actor–critic의 4×7×4×4 = **448개**.
- 112개 공통 prefix + mass/split 224개 continuation + source 4개 + replay 112개 + closed 112개 = **564개 실행 노드**. Prefix는 두 최종 궤적이 공유하며 중복 독립 실험으로 세지 않는다.

## Q를 직접 변경

행동은 [-1,1], temperature는 τ=0.25. c=(-0.6,0,0.6), h=0.1이다.

$$f_t(a)=\sum_k \rho_{k,t}\mathcal N(a;c_{k,t},0.1^2),\quad Q_t(a)=0.25\log f_t(a),\quad \pi_t^\star(a)=\frac{f_t(a)}{\int_{-1}^1f_t(x)dx}.$$

세 균등 mode를 20,000 updates 학습한다. 같은 method/size/seed의 actor, Adam, training RNG를 그대로 복사해 mass와 split으로 분기한다. 다른 방법끼리 학습된 actor를 공유하지 않는다. 초기 mean network 파라미터는 seed별로 동일하고 fixed sigma 0.1/0.5는 해당 출력을 명시적으로 override한다.

**Step 경계:** t는 1부터 시작하는 actor update다. 1..20000은 공통 target이며 변경은 20001번째 update부터 적용한다.

| 종류 | Update | 변경 |
|---|---|---|
| mass | 1..20000 | weights=(1/3,1/3,1/3) |
| mass | 20001..25000 | weights=(0.6,0.2,0.2) |
| mass | 25001..30000 | weights=(0.2,0.2,0.6) |
| mass | 30001..35000 | 균등 weights 복구 |
| split | 1..20000 | d=0 |
| split | 20001..22000 | d=0.15(t−20000)/2000 |
| split | 22001..25000 | d=0.15 |
| split | 25001..27000 | d=0.15(27000−t)/2000 |
| split | 27001..35000 | d=0 |

Split에서는 각 c_k를 c_k−d, c_k+d로 나누고 Gaussian 6개에 1/6씩 배분한다. d=0은 동일 Gaussian들이 겹쳐 원래의 3-mode density와 정확히 같고, d=0.15에서는 여섯 peak가 된다. 폭·총질량·temperature를 바꾸지 않는다. Bounded normalization 이후 실제 basin 질량은 명목 mixture weight와 구분한다.

## 학습되는 critic

$$s,a\in[-1,1],\quad s'=a,\quad s_0=0,$$
$$g(a)=\sum_{c\in\{-0.6,0,0.6\}}\exp[-(a-c)^2/(2\cdot0.1^2)],\quad r(s,a)=g(a)-0.25(a-s)^2.$$

Reward·transition은 stationary다. 200-step timeout 뒤 state를 0으로 reset하지만 replay의 next_state에는 reset 전 a를 저장하고 bootstrap을 유지한다. 5K uniform warmup 이후 35K actor/critic updates, 총40K environment steps다. 이후 추가 uniform exploration은 없다.

$$a'\sim\pi_\theta(\cdot|s'),\quad y=r+0.99\min_\ell\bar Q_\ell(s',a'),\quad Q_{extract}=(Q_1+Q_2)/2.$$

원래 v5의 VectorCritic과 OptiQDIME.update_critic/soft_update를 직접 호출한다. Reward-only TD, target twin min, current actor의 stochastic next action, Polyak coefficient .005다. Actor teacher는 업데이트된 live twin mean을 사용한다. Critic update → target critic update → actor update 순서를 유지한다.

**Source 4 runs:** seed별 v5 learned-sigma Sinkhorn, 16×64, actor/critic batch256으로 실제 critic 궤적을 만든다. 매 critic update 직후 Q_extract(0,a)를 [-1,1]의 endpoint-inclusive 8193개 grid에 기록한다. 시간축 보간·건너뛰기는 없다. Action query만 선형 보간하며 16385개 독립 grid와 interpolation L∞, bin TV, backup 차이를 검사한다(각1e−4 미만). 검증 실패는 reference failure로 남기고 해당 run을 중단한다.

**Replay 112 runs:** 같은 seed의 source에서 매 update의 동일 Q(0,a)를 받아 actor만 새로 학습한다. Actor state batch1. 자신의 행동이나 critic이 source를 바꾸지 않는다. Q source는 Sinkhorn v5에서 생성된 하나의 조건이며 모든 가능한 Q 궤적을 대표한다고 주장하지 않는다.

Reference grid와 Q 궤적 기록에만 JAX `default_matmul_precision=highest`를 적용한다. TF32 연산의 양자화 계단이 grid 보간 오차처럼 나타나는 것을 막기 위한 수치 설정이다. Actor/critic 학습 경로는 v5의 원래 정밀도를 유지하며 reference와 학습 정밀도의 Q 차이도 별도 저장한다.

**Closed 112 runs:** 각 방법이 자신의 환경 데이터와 critic을 학습한다. Critic replay batch256, actor는 그 batch의 첫 state 하나를 사용한다(각 replay index는 uniform sample). Actor batch1은 큰 N×M에서 일관된 1D 진단을 위한 명시적 설정이며, MuJoCo 기본 batch256과 구분한다. Stochastic 10 episodes×200 steps return을 기록한다. 진단 probe state는 모든 run에서 s=0이다.

## 공통 구현 설정

| 항목 | 값 |
|---|---|
| Source | v5 snapshot a5d5e281bcb10b5c31557073d95e03a1c5f2af4f + 검증된 Direct GMM loss |
| Actor / critic | 각각 256×2 GELU; norm/dropout 없음 |
| Latent | 매 update마다 새로운 Normal(0,1), action dimension1 |
| Actor σ | learned 초기0.5, log σ clip [−5,1]; fixed는0.5 또는0.1 |
| Fixed σ 적용 | loss, proposal, 실제 행동 수집, TD next action, 평가 모두 동일 |
| Proposal | 현재 N개 conditional Gaussian mixture, IID component sampling, 총 M개 |
| Density correction | w=softmax(Q/.25−log q_F(a)), beta1; tanh Jacobian 포함 |
| Teacher σ floor | proposal에만 최소0.05 |
| OT source | tanh μ; 원래 student epsilon RNG draw는 유지 |
| Cost | raw squared action distance |
| Sinkhorn | epsilon0.1, iterations100; 원래 row normalization 유지 |
| Exact OT | 1D monotone coupling, CPU float64 mass intervals, float32 plan/NLL |
| GMM | marginal mixture NLL, OT cost/Sinkhorn/plan 미계산 |
| Optimizer | Adam3e−4, (0.9,0.999), eps1e−8, clipping 없음 |
| 추가 기법 | EMA/history, guard, annealing, adaptive weight 보정 없음 |
| Replay capacity | 1M; 실제40K 데이터만 채워짐 |

Conditional NLL의 teacher action Jacobian은 parameter-independent이므로 원래 v5와 같이 생략한다. 공통 평가 marginal NLL에는 해당 Jacobian을 포함한다. 생성된 pre-tanh u를 보존하며 atanh(tanh(u))로 복원하지 않는다.

## 관찰·저장

- 매 update: loss, gradient norm, ESS/wmax, sigma, mean 이동량, between/within variance, OT P와 실제 R/N의 marginal residual. RL은 critic metric과 수집 transition도 기록.
- 기본200 updates마다, 변경 경계 ±100 updates 구간은20 updates마다: **32768개의 독립 latent·noise pair**에서 실제 action을 생성하여512-bin histogram 저장. 학습 RNG와 평가 RNG를 분리한다. 시간축 비교는 common evaluation RNG를 쓰며 최종에는 별도 독립 RNG도 평가한다.
- 같은 probe state와 critic에서 별도 proposal을 재표집하여 unweighted proposal / weighted teacher / actor histogram을 함께 기록. 각자 방식의 proposal임을 명시한다.
- 고정 evaluation latent의 mean/sigma를 시간에 따라 비교한다. Training row는 매번 새 latent이므로 영구 component로 부르지 않는다.
- 선택시점: 1,1K,10K,19,980,20K,20,001,20,020,21K,22K,24,980,25K,25,001,25,020,26K,27K,29,980,30K,30,001,30,020,35K. 같은 teacher에서 update 전후 marginal NLL·mean·sigma와 training teacher를 기록한다.
- Probe assignment raw: OT P, R, 실제 loss mass R/N, GMM wγ, row mass/variance, 원본·정렬 index를 저장한다. Source batch256 raw는 첫 state 한 개만 보관하고 실제 학습 batch 크기를 따로 표시한다.
- Actor/Adam/RNG, critic/target/Adam, 전체 유효 replay·sampling RNG, env state·episode 위치를200 updates마다 원자적으로 checkpoint한다. Prefix checkpoint는 읽기만 한다.

## 지표 해석

Histogram TV = 1/2 Σ_bin |actor mass−target mass|. Basin-mass TV는 같은 계산을 target peak 사이 valley로 나눈 구간에 적용한다. 예를 들어 두 basin의 목표가 (0.5,0.5), actor가 (0.8,0.2)이면 TV=0.3이다. Basin 질량이 맞아도 valley를 채운 넓은 Gaussian일 수 있으므로 histogram TV와 density를 주 지표로 삼는다.

Learned Q가 flat하거나 multimodal하지 않은 구간도 그대로 보여준다. Target의 peak 개수나 경계가 바뀌면 basin별 번호를 영구적인 mode identity로 연결하지 않는다. Full histogram은 고정 bin을 사용하므로 시간에 따라 직접 비교할 수 있다.

추가 진단: proposal/teacher TV, actor–teacher TV, common marginal NLL, E_actor Q−E_target Q, component usage, sigma와 mean 다양성. Teacher와 actor의 TV 차이를 인과적 오차 분해 등식으로 해석하지 않는다.

변경 후 histogram-TV 누적량과 stationary 복귀 후 회복 시간을 보고한다. 회복 기준은 해당 run의19K..20K 평균 TV+0.05 이하가3회 연속 관찰되는 최초 시점이다. Mass는30K, split은27K부터 계산하고35K까지 미회복은 censored로 남긴다. Fixed σ0.5는 표현력 한계가 있으므로 절대 오차와 변화 전 대비 추가 오차를 함께 제시한다.

## 실행·보관

계산: login4:/lustre/hobbit9882/OptiQ-nonstationary-q-20260917.
보관: dildata:/data1/heejoonorm/OptiQ/studies/20260917_nonstationary_q.
코드·설계·tasks·manifest를 먼저 저장하고 source hash가 일치하는 검증 뒤에만 main을 시작한다. GPU1개당 CPU2개, RAM24GB. 같은 서버의 다른 실험을 취소하거나 변경하지 않는다. 실패·중단을 완료로 세지 않고 성능 기반 early stopping을 하지 않는다.

검증 → prefix → mass/split → source → replay → closed 순서. Source 준비4개와 validation은448개 결과에 포함하지 않는다. 보고서 MD/HTML에는 표본 수, step, seed, exact/sampled reference 구분과 미완료 여부를 먼저 표시한다.

SLURM에는 stage별 array를 등록하며 최대16 GPU를 병렬 요청한다. 실제 동시 실행 수는 가용 자원과 스케줄러 우선순위에 따른다. GPU1개당 CPU2개,6시간 time slice를 사용하며 종료 예고 시 checkpoint를 저장하고 같은 array task로 재개한다. 검증에서 GPU 장치가 노출되지 않은 node24와 이전 작업에서 확인된 비정상 노드(cs-gpu-01, node14, node23, node31, node35, node40)는 제외했다. 비선점 big_qos를 사용한다.

dildata의 tmux `optiq-nonstationary-sync`가3분마다 결과를 직접 가져온다. Collector 전용 SSH key는 dildata에만 저장하며 login4에서 해당 실험 폴더의 읽기 전용 rsync로 제한했다. 완료 run은 artifact hash까지 확인해 `BACKUP_VERIFIED.json`을 만든다. 전체 제출 목록은 `SUBMISSION.json`, 수치 검증 결과는 `VALIDATION_0.json`부터 `VALIDATION_3.json`, 백업 현황은 `STORAGE_SYNC_STATUS.json`에 기록한다.
