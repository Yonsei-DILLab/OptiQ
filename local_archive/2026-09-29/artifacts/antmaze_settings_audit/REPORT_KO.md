# AntMaze 설정 및 스케일 감사 — 2026-09-23

**현재 가장 강한 증거는 OptiQ에서 후보 행동의 Q 차이가 temperature에 비해 너무 작아 정책 개선 가중치에 거의 반영되지 않는다는 것이다. Sigma 상한도 실제로 활성화되어 있지만, 전체 행동 분포가 좁거나 거의 결정적이라는 설명은 측정과 맞지 않는다. MFPO는 오히려 temperature가 매우 작아진 다른 양상을 보인다. 따라서 모든 실패를 하나의 sigma 설정 탓으로 묶을 수 없다.**

이 결론은 설정 파일만 보고 추측한 것이 아니다. 기존 W&B 원시 history를 읽고, 보존된 최종 checkpoint를 CPU에서 다시 불러 각 정책의 replay에서 128개 상태를 뽑아 상태당 64개 행동의 Q·분산·가중치를 계산했다. RND는 정책마다 8,192개 replay transition에서 별도 재평가했다. optimizer update, 환경 학습, 새 실험 등록은 하지 않았다.

**분석 범위와 출처**

- 대상: 공식 환경으로 바꾼 뒤의 `antmaze-upstream-sparse256-nativebudget-s0-20260923`, sparse + NovelD 0.01, seed 0.
- 학습 source: `0ebd8d26c711d79723f457343b36eb8788bd87b8`.
- DDiffPG 원본: `7edd06c4799abbab0f8fa534c21deb56253b018e`. 가져온 159개 파일을 manifest SHA256과 다시 대조했고 모두 일치했다.
- MFPO 원본: `d8b3977d29d4ef2d315e871337e5826f2eb79eb2`. `configs/mfpo_config.py`의 범용 설정을 사용했다. 공개 `train_online.py` 기본 환경은 `Ant-v3`이며, 이것을 논문의 AntMaze 전용 설정으로 확인한 것은 아니다.
- 완료된 OptiQ/MFPO v1·v2: 각 3,008,256 interactions, 93,752 learner/RND updates. 각 학습 seed는 0 하나다.
- DIPO v1–v4는 학습 중 목표 도달을 기록한 뒤 수치 오류로 중단했다. SAC 네 개 및 v3/v4 OptiQ/MFPO는 대기열 보류로 시작하지 않았다. 그러므로 이 캠페인을 “SAC까지 모두 충분히 학습했지만 실패했다”고 해석할 수 없다.
- 이후 dense + NovelD OFF의 328,192-interaction 짧은 probe는 별도 실험이며, 여기의 3M sparse 실패 분석과 혼합하지 않았다.

**원본 repository에서 실제로 선택되는 설정**

아래는 YAML의 이름뿐 아니라 runtime 분기와 네트워크 생성자까지 따라간 값이다. DIPO YAML의 `act_class: TanhMLPPolicy`는 최종 actor 구조를 뜻하지 않는다. `ActorCriticBase`가 방법 이름을 보고 DiffusionPolicy로 교체한다.

| 항목 | DDiffPG 본체 | 원본 DIPO baseline / 우리 sparse DIPO | 원본 SAC baseline / 우리 SAC 구성 | 우리 OptiQ | 우리 MFPO |
|---|---|---|---|---|---|
| 병렬 환경 | 256 | 256 | 256 | 256 | 256 |
| Batch | 4096 | 4096 | 4096 | 4096 | 4096 |
| 수집 / learner update | 256개 / 8회 | 동일 | 동일 | 동일 | 동일 |
| 실제 update / transition | 1/32 | 동일 | 동일 | 동일 | 동일 |
| Random warmup | 500 vector steps = **128,000 interactions** | 32 × 256 = **8,192** | 8,192 | 8,192 | 8,192 |
| Actor | diffusion 5 steps, 본체 MLP 1024–512–256, Mish + time embedding MLP | 동일 | 512–256–128, ELU, tanh Gaussian | 256–256 GELU, box-truncated Gaussian mixture, random z, N=M=64 | 256×3 GELU + LayerNorm, MeanFlow 2 steps |
| Critic | 512–256–128 ELU, distributional twin Q, 모드별 critic | 512–256–128 ELU, distributional twin Q | 512–256–128 ELU, **scalar** twin Q | 256×2 GELU, scalar twin Q | 256×3 GELU + LayerNorm, distributional twin Q |
| Categorical support | [0,5], 51 atoms, 간격 0.1 | 동일 | 해당 없음 | scalar이므로 config의 v_min/v_max는 support가 아님 | **[-1600,1600], 101 atoms, 간격 32** |
| Actor / critic LR | 3e-4 / 5e-4 | 동일 | 동일 | 3e-4 / **3e-4** | 3e-4 / **3e-4** |
| Optimizer | AdamW | AdamW | AdamW | Adam | Adam |
| Gradient norm clip | 1 | 1 | 1 | 없음 | 없음 |
| Target critic τ | **0.05** | 0.05 | 0.05 | **0.005** | **0.005** |
| γ / n-step | .99 / 1 | 동일 | 동일 | 동일 | 동일 |
| Replay | 2000 trajectories 및 모드 분리 | 1M transitions | 1M transitions | 1M transitions | 1M transitions |
| 보상/관측 정규화 | 끔 | 끔 | 끔 | 끔 | 끔; hidden LayerNorm은 별개 |
| NovelD | 0.01, normalize=false | 동일 | 동일 | 동일 원본 IntrinsicM | 동일 원본 IntrinsicM |
| Actor의 Q 사용 | Q-gradient로 target action을 20번 개선, action LR .03 | 동일 | reparameterized Q-gradient + entropy | `softmax(Q/.25 − log q)` teacher를 marginal GMM NLL로 학습 | adaptive-temperature Q 가중치 + kernel/density 항으로 flow 학습 |
| 외부 행동 잡음 | 환경별 σ=.05… .6, decay 없음 | 동일 | 별도 mixed noise 없음; Gaussian 정책에서 샘플 | DACER, 초기 .027 → 최종 약 **.01998** | 별도 mixed noise 없음; flow prior 샘플 |
| 정책 내 확률성 | DDPM 초기/역확산 잡음 | 동일 | pre-tanh log σ 허용범위 **[-5,5]**, 학습 | conditional log σ **[-5,-1]**, 초기 -1, latent mean도 랜덤 | Gaussian prior에서 flow 생성; OptiQ 같은 conditional σ 상한 없음 |
| Entropy 계수 | SAC식 계수 없음 | 없음 | α 초기 1, 자동 조절, LR .005, target H=-8 | actor T=.25 고정; DACER target proxy=-7.2는 외부 noise만 조절 | α 초기 .01, 자동 조절, LR3e-4, target H=-4 |
| Critic entropy backup | 없음 | 없음 | 있음 | **없음** | 있음 |

DDiffPG/DIPO의 target-action smoothing `σ=.8, clip=±.2`는 critic backup에 쓰이는 별도 잡음이다. 환경에 넣는 탐색 잡음 `.05… .6`과 구분해야 한다. DDPM 초기 latent의 표준편차 1, SAC의 pre-tanh σ, OptiQ의 bounded-action conditional σ를 그대로 같은 척도의 탐색량으로 비교해서도 안 된다.

공통 batch와 update 비율은 원본 **baseline**에 맞게 실행됐다. OptiQ config 내부 `utd=1`은 외부 루프가 8번 호출하는 1회 update를 뜻한다. 실제 checkpoint counter도 93,752로 맞는다. “OptiQ/MFPO만 transition당 UTD1로 학습했다”는 상황은 아니다. 다만 원본 DDiffPG 본체의 warmup 128k와는 다르며, SAC/DIPO baseline에 맞춘 선택이었다.

기본 maze budget은 v1/v2 3M, v3 4M, v4 5M이며 원본 global counter는 warmup을 제외하고 `> max_step`에서 종료한다. baseline 기준 실제 총 interactions는 각각 3,008,256 / 4,008,448 / 5,008,384다. seed0, 평가250k, 최종 checkpoint 저장 등은 사용자 요청에 따른 실행 프로토콜 차이다.

원본 코드 근거: [default.yaml](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/cfg/default.yaml), [공통 optimizer 설정](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/cfg/algo/actor_critic.yaml), [DIPO 설정](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/cfg/algo/dipo_algo.yaml), [SAC 설정](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/cfg/algo/sac_algo.yaml), [runtime actor 선택](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/algo/ac_base.py:26), [실제 diffusion network](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/models/diffusion_mlp.py:47), [SAC log σ 범위](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/models/mlp.py:87).

논문과 공개 코드도 완전히 같지는 않다. 논문은 모든 baseline에 distributional RL, optimizer Adam이라고 기술하지만, 이 commit의 SAC는 scalar `DoubleQ`, 공통 optimizer는 AdamW다. 따라서 위 표는 논문 문구를 그대로 옮기지 않고 **실제로 실행한 공개 코드**를 기준으로 했다. [DDiffPG 논문 §5.1·Appendix E](https://arxiv.org/html/2406.00681v1#A5)

**1. OptiQ: reward 크기보다 같은 상태에서의 ΔQ/T가 결정적이다**

실제 teacher 식은 다음과 같다. 현재 RL source에는 과거 toy GMM 논의의 `Q 앞에 .25를 곱하는 보정`이 없다.

`w_i = softmax(Q(s,a_i)/0.25 − log q(a_i|s))`

Q에 같은 상수를 더해도 softmax는 바뀌지 않는다. 따라서 Q 평균이 .38이라는 것보다, **동일 상태의 후보 행동 사이 Q 차이**가 중요하다. 최종 checkpoint를 동일한 actor/critic 구현으로 다시 평가한 결과:

| 측정 | OptiQ v1 | OptiQ v2 |
|---|---:|---:|
| 후보 Q 표준편차, 상태별 계산 후 평균 | 0.000825 | 0.001026 |
| Q/T 표준편차 | 0.003302 | 0.004103 |
| -log q 표준편차 | 0.9636 | 0.9617 |
| density 항 표준편차 / Q 항 표준편차 | **292배** | **234배** |
| Q만으로 가중할 때 ESS / 후보64개 | **63.999** | **63.999** |
| Q 항 제거 전후 teacher 가중치 TV 거리 | **0.137%** | **0.176%** |

TV는 `0.5 × Σ|w_full−w_Q제거|`를 상태별 계산해 평균낸 값이다. 두 항의 표준편차 비교뿐 아니라, **동일 후보에서 Q 항을 실제로 제거해도 teacher 가중치가 거의 달라지지 않는 것**을 확인했다. 여기서 density correction이 잘못 구현됐다는 결론은 나오지 않는다. Q가 temperature에 비해 평평하면 uniform-action 기준 Boltzmann target 자체가 거의 균일해지고, importance correction은 그 target을 근사하는 역할을 한다.

같은 checkpoint에서 T만 바꿔 재계산한 가중치 TV는 v1/v2에서 T=.01이면 3.45%/4.45%, T=.005면 6.94%/8.98%, T=.001이면 33.39%/41.58%다. **이는 학습 성능 실험이 아니며 .001이 최적이라는 뜻이 아니다.** 기존 10k-update temperature 실험은 T=.005까지 낮춰도 성공하지 못했다. 더 낮은 T가 보상과 관련 없는 critic 오차를 확대할 가능성도 함께 검증해야 한다.

최종 checkpoint의 두 critic 간 동일 상태·후보 Q 상관은 v1 .936, v2 .952였다. 후보의 작은 차이에 두 critic이 상당히 동의하지만, 같은 replay/TD target으로 학습하므로 이것만으로 실제 장기 보상에 대한 정확성을 보장하지 않는다.

**2. Sigma 상한은 실제로 걸린다. 그러나 전체 행동 랜덤성이 부족한 상태는 아니다**

두 OptiQ 모두 conditional σ=exp(-1)≈.368에 100% 붙었다. raw log σ도 sampled states/actions에서 전부 -1보다 컸다: v1 평균 -.858, 범위 [-.978,-.660]; v2 평균 -.812, 범위 [-.939,-.615]. 따라서 단지 로그 표시상 상한과 같아 보이는 것이 아니라 hard clip이 실제 활성화되어 있다. 이 지점에서 clipped log σ의 raw head에 대한 국소 미분은 0이다. 공유 hidden layer가 바뀌면 raw 출력은 다시 움직일 수 있으므로 영구적으로 복구 불가능하다는 뜻은 아니다.

하지만 σ는 latent 하나를 조건으로 한 Gaussian의 폭이다. z가 바뀌면 μ도 크게 변한다.

| 최종 정책의 같은 상태 내 행동 분산 | v1 | v2 |
|---|---:|---:|
| OptiQ μ-only 좌표 RMS 표준편차 | .470 | .469 |
| OptiQ full-policy 좌표 RMS 표준편차 | **.521** | **.520** |
| MFPO direct-policy 좌표 RMS 표준편차 | .177 | .183 |
| 참고: [-1,1] 균등분포의 좌표 표준편차 | .577 | .577 |

OptiQ는 이미 행동 공간에서 상당히 넓게 퍼져 있다. σ를 늘리면 달라질 여지는 있지만 “행동이 너무 비슷해서 못 움직였다”는 주된 설명은 뒷받침되지 않는다. 독립적인 관절 행동의 분산이 큰 것과 여러 step에 걸쳐 유용한 보행·경로를 만드는 것은 다르다. 상한을 늘리는 것만으로 이미 약한 Q 기반 선택이 강해지지는 않는다.

DDiffPG/DIPO의 환경별 외부 noise는 .05… .6이고 전체 worker 기준 RMS 약 .362다. OptiQ의 DACER 외부 noise는 최종 약 .01998로 이 추가 noise만 비교하면 약18배 작다. **이 비율을 전체 정책 탐색량의 18배 차이로 해석하면 안 된다.** OptiQ에는 위에 측정한 latent 및 conditional 확률성이 이미 존재한다. DACER의 GMM entropy proxy는 약5.84~5.85, target은 -7.2여서 controller는 외부 noise를 줄였다. 이 proxy는 정확한 정책 entropy가 아니며 state coverage나 보행 능력을 측정하지 않는다.

OptiQ 구성의 출발점은 실제로 `benchmark=ant` + `mujoco_v5`다. T=.25, log σ[-5,-1], DACER noise_scale=.1, scalar256×2 critic, τ=.005가 남아 있었다. **AntMaze 환경을 원본으로 바꾼 것과 그 보상 스케일에 OptiQ를 조정한 것은 별개의 작업이었다.**

**3. MFPO: 같은 실패지만 OptiQ와 다른 가중치 스케일**

MFPO의 초기 temperature는 .01이지만 고정값이 아니다. 최종 자동 조절 값은 v1 `1.47e-5`, v2 `1.83e-5`였고, 마지막100k logged batch의 entropy는 target -4 근처였다.

| 측정 | MFPO v1 | MFPO v2 |
|---|---:|---:|
| 같은 상태의 후보 Q 표준편차 | .000325 | .000384 |
| Q/temperature 표준편차 | **22.08** | **20.95** |
| 공통 진단용 Q-only ESS / 64 | **1.40** | **1.44** |
| critic support | [-1600,1600], 101 atoms | 동일 |
| 0 atom의 평균 확률질량 | 80.5% | 81.3% |
| -32,0,+32 세 atom의 질량 | 97.7% | 97.9% |

OptiQ는 Q-only 가중치가 거의 균일하지만 MFPO는 아주 작은 Q 차이도 크게 확대한다. 이 ESS는 두 방법을 비교하려고 같은 방식으로 계산한 진단이다. MFPO 실제 actor 가중치에는 kernel·log-density 항과 16개 policy/32개 clean proposal이 있으므로, “실제 학습이 64개 중 1.4개만 사용한다”고 단정할 수 없다.

범용 MFPO support의 간격32는 DIPO의 .1보다 320배 크고, 실제 sparse 학습에서 받은 NovelD는 약 .004였다. 이 값 범위는 세밀한 TD 차이를 분포 확률의 작은 변화로 학습하게 만드는 **검토할 만한 스케일 차이**다. 다만 categorical distribution의 기대값은 연속값을 표현할 수 있으므로 Q가 32단위로 반올림된다는 뜻도, 이것이 실패 원인으로 확정됐다는 뜻도 아니다. MFPO는 entropy backup도 있으므로 DIPO의 비음수 support[0,5]를 그대로 복사하는 것이 적절하다고 단정할 수 없다.

**4. NovelD 0.01은 원본과 같다. 다만 보너스가 존재한다고 공간 탐색을 강하게 유도하는 것은 아니다**

현재 수식은 원본 그대로 `0.01 max(n(s')−0.5n(s),0)`다. n은 RND feature error의 L2 norm이며, predictor 학습 loss는 MSE다. xy Fourier10 bands에 나머지 관측을 붙인 69차원 입력, normalize=false, AdamW1e-4, gradient clip1을 쓴다. 좌표만의 신규 방문 여부를 측정하지 않으며 이 구현에는 방문 상태를 episode당 한 번만 보상하는 gate가 없다. [원본 구현](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/utils/intrinsic.py:23)

완료한 OptiQ/MFPO의 외재보상은 전부0이었다. 최종 replay와 RND로 다시 계산한 OptiQ 보너스 평균은 v1 .003813, v2 .003631이다. 입력을 가상으로 `(s,s)`로 바꿔도 평균 .003822/.003633이었다. 같은 상태라 해도 `max(n−.5n,0)=.5n`이므로 RND 오차가 남아 있으면 양의 보너스가 나오기 때문이다. 실제 transition에서 xy 이동량과 보너스의 상관은 각각 .124/.089였다.

**이 검사는 실제 환경에서 정지 정책을 실행했다는 뜻도, 학습 보너스의 100%가 정지에서 왔다는 뜻도 아니다.** 평균 유사성과 상관만으로 인과 관계를 확정할 수 없다. 다만 .004 전체를 “새로운 공간으로 나아가는 추가 보상”으로 해석하면 안 된다는 근거다. RND가 익숙해지면 그 오차와 보너스는 줄어들 수 있다. 원본 DDiffPG도 같은 수식이므로 이것 자체가 우리 이식의 오류는 아니다.

마지막100k OptiQ logged batch 평균은 v1 reward .003838, Q .3823; v2 reward .003655, Q .3685였다. 단순한 정상상태 근사 `reward/(1−.99)`는 .3838/.3655로 관측 Q와 비슷하다. 이는 critic이 전혀 보상을 받지 않은 것이 아니라, 주로 반복적으로 얻는 작은 intrinsic return의 수준을 학습하면서 **행동 간 추가 가치 차이는 매우 작게 남은 상황**과 일치한다. 이 근사는 이동·종료를 생략한 설명이지 정확한 Bellman 검증은 아니다.

NovelD 계수를 키우면 이 보너스 전체가 커진다. 같은 상태에서 생기는 보너스도 함께 커지며, 목표 보상10/20과 intrinsic reward의 상대 비중도 바뀐다. 따라서 “계수를 키우면 새 공간만 더 선호한다”는 보장은 없다. Q가 학습된 뒤 reward를 동일 배수로 확대하는 것과 T를 줄이는 것은 단순 Boltzmann 가중치 관점에서 비슷할 수 있지만, NovelD만 키우는 것은 외재보상과의 비율까지 바꾸므로 두 실험은 동일하지 않다.

**5. DIPO가 작은 보상에서도 움직였던 이유로 검토할 차이**

DIPO는 고정 T의 Boltzmann 가중치를 쓰지 않고, replay의 target action을 Q-gradient로 20번 개선한 뒤 diffusion actor를 회귀시킨다. 외부 mixed noise, 더 큰 1024–512–256 actor, τ=.05, critic LR5e-4, 좁은 비음수 support가 결합되어 있다. 이들은 작은 intrinsic 신호를 행동에 반영하는 방식과 지속적인 탐색에 차이를 줄 수 있다. **어느 한 항목이 성공을 만들었다고 아직 분리해 입증하지는 않았다.**

DDiffPG 본체는 여기에 성공 trajectory clustering, 탐색 critic과 경로별 critic, trajectory buffer를 추가한다. 본체의 다중 경로 유지 능력을 단순 sigma 차이로 설명할 수 없다. 특히 탐색용 critic에는 intrinsic reward를, 다른 mode critic에는 환경보상+intrinsic을 주는 분기가 있다. 우리는 현재 DIPO baseline과 비교 중이며 DDiffPG 본체를 실행한 것은 아니다. [본체 update 로직](/Users/yunheechan/Documents/ChatGPT/OptiQ/antmaze/ddiffpg/algo/ddiffpg.py:178)

과거 DIPO의 중단은 별도의 수치 문제다. 원본 C51 projection에서 terminal reward10/20이 support상한5에 몰리면 float32 합산 확률이 1을 아주 조금 넘을 수 있고 BCE assert가 난다. 합성 case에서 재현했으며 실제 실패 minibatch는 남아 있지 않아 정확한 trigger를 확정하지 않았다. 이후 adapter의 안정화와 CPU/GPU 수치 검증, dense probe 10k updates는 통과했다. 이 버그는 OptiQ/MFPO의 성공0이나 sigma 범위를 설명하지 않는다.

**원인에 대한 판단과 다음 검증 순서**

| 판단 | 근거 수준 |
|---|---|
| OptiQ의 현재 Q/T는 actor의 행동 우열을 거의 반영하지 못한다 | 최종 checkpoint에서 Q 제거 가중치 비교로 직접 확인 |
| OptiQ hard sigma cap이 활성화되어 있다 | raw/clipped log σ 및 saturation 100%로 확인 |
| 그래서 sigma를 늘리면 길을 찾는다 | 미검증. 전체 행동 분산은 이미 큼 |
| NovelD가 빠졌거나 256env에서 업데이트 수가 잘못 줄었다 | 코드·counter·replay·RND history와 맞지 않음 |
| NovelD가 현재 방문 상태에서 공간 이동을 충분히 구별하지 못한다 | 검토할 근거가 있음. 평균 보너스와 ΔQ 차이, xy 상관이 작음 |
| MFPO도 OptiQ와 똑같이 T가 너무 높아 실패했다 | 최종 adaptive-temperature 측정과 맞지 않음 |
| MFPO의 범용 support/entropy 제어가 sparse reward에 부적합하다 | 설정·로그상 원인 후보. 재학습 대조 실험 필요 |
| 하나의 hyperparameter가 모든 실패의 원인이다 | 현재 자료로 입증 불가 |

우선 원본 SAC와 안정화된 원본 DIPO를 같은 sparse profile로 충분히 학습한 기준 결과가 필요하다. OptiQ는 **Q/T 영향과 sigma cap을 독립적으로 바꾸는 대조**가 원인 분리에 적절하다. 후보를 넓히는 변경과 행동 선택을 강하게 하는 변경을 동시에 여러 개 넣으면 어느 요인이 유효했는지 알 수 없다. MFPO는 별도로 실제 soft-TD target 범위에 맞는 critic support와 entropy/temperature 동작을 점검해야 한다. 초기 random warmup, τ, critic LR, 모델 크기도 원본과의 차이지만 이번 자료에서 1순위 원인으로 확정되지는 않았다.

최적 T나 sigma는 이번 계산만으로 선정하지 않았다. 긴 실험을 추가로 등록하거나 기본값을 수정하지 않았으며, dense+NovelD OFF 본 캠페인도 아직 등록하지 않았다.

![체크포인트 스케일 진단](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/scale_diagnostics.png)

**재현 자료**

[통합 수치·측정 정의](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/analysis.json), [forward 진단 코드](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/inspect_saved_policy.py), [NovelD 진단 코드](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/inspect_saved_noveld.py), [원시 history 추출 코드](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/read_history.py), [보존된 실제 학습 소스](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_settings_audit/source_snapshot/manifest.json).

샘플링은 각 정책 자기 replay에서 수행했으므로 방법 간 정확히 동일한 상태 집합은 아니다. 모든 결과는 단일 학습 seed이며, 새 temperature로 재가중한 수치는 frozen checkpoint의 내부 동작 진단일 뿐 학습이나 성공률의 반사실 결과가 아니다. 기존 CSV의 간헐적인 상세 진단 로그와 이번 최종 checkpoint 측정은 시점·미니배치가 달라 수치가 약간 다를 수 있다.
