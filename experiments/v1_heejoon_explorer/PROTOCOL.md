# v1_heejoon_explorer — legacy explorer / EMA evaluator

## 목적과 역할

Explorer가 critic 변화에 빠르게 반응하되, 그 순간의 행동 변화가 TD continuation과
evaluation에 동시에 반영되는 경로를 EMA evaluator로 완화하는 실험이다.
안정성/성능 개선은 아직 가설이다. Parameter EMA는 분포 거리나 성능의 단조성을 보장하지 않는다.

| 역할 | 사용 policy |
|---|---|
| 환경 데이터 수집 | Explorer (첫 5K는 기존 uniform warmup) |
| OT source / KDE 중심 / 후보와 importance weights | Explorer, 기존 legacy update 그대로 |
| Actor 학습 | Explorer만 rowwise-argmax MSE gradient update |
| TD 다음 행동 | Evaluator의 stochastic latent action |
| TD Q | Target twin scalar critic의 min |
| Actor extraction Q | Live twin scalar critic의 mean, legacy 그대로 |
| Evaluation | Evaluator만, stochastic latent; explorer 평가 return과 섞지 않음 |
| Evaluator 학습 | Explorer gradient update 직후 parameter EMA 한 번 |

\[
a_t\sim\pi_{\rm exp}(\cdot\mid s_t),\qquad
a'\sim\pi_{\rm eval}(\cdot\mid s'),
\]
\[
y=\operatorname{stopgrad}[r+0.99(1-d)\min_k Q_{\bar\phi_k}(s',a')],
\qquad L_Q=\sum_{k=1}^2\operatorname{mean}(Q_{\phi_k}(s,a)-y)^2.
\]
\[
\theta_{\rm eval}\leftarrow0.995\theta_{\rm eval}+0.005\theta_{\rm exp,new}.
\]

Evaluator는 별도 loss/gradient update가 없다. Twin-min은 TD에만 쓰며 evaluator
최적화 objective가 아니다. Density/KL trust region도 넣지 않는다. 초기 evaluator는
explorer와 같은 파라미터이다. EMA 0.005의 반감기는 약 138 actor updates다.

## 기본 설정과 기존 legacy 대비 차이

Root `configs/mujoco_setting.yaml`의 scalar/no-anchor legacy 설정을 상속한다.
v5 conditional Gaussian이나 Direct GMM actor를 사용하지 않는다.

| 항목 | 설정 |
|---|---|
| Actor | a=clip(G(s,z),-1,1), standard-normal z; GELU 256×3 |
| N / M | 16 source / 64 random candidates; center당 4, anchor 없음 |
| KDE | Truncated Gaussian std 0.2, local clip 0.5 |
| Weight | Q/0.25 - log q; fixed full density correction beta=1 |
| OT | Mean-normalized squared cost; Sinkhorn epsilon 0.05, 30 iterations |
| Loss | Row-argmax candidate에 대한 raw actor output MSE |
| Critic | Scalar twin 256×3 GELU, BN 없음; reward-only; target tau 0.005 |
| UTD / actor delay / batch | 1 / 1 / 256 states |
| Adam | Actor / critic learning rate 3e-4; beta1 0.9, beta2 0.999 |
| Length | 각 1M env steps, uniform warmup 5K |
| Evaluation | 첫 step 및 5K마다 stochastic evaluator 10 episodes |
| Seeds | 0,1,2,3, 환경별 순차 실행 |
| W&B | OptiQ/legacy_explorer, algorithm_name=v1_heejoon_explorer |
| 환경/GPU | Ant-v4 → GPU0, Humanoid-v4 → GPU1, HalfCheetah-v4 → GPU2; GPU3 비워둠 |

핵심 변경은 policy_tau 1→0.005, evaluation policy를 evaluator로 전환,
TD action의 추가 truncated perturbation(std0.2/clip0.5) 제거다. 마지막 변경은
사용자가 제시한 `a'~pi_eval` 식을 그대로 따르기 위한 것이다. 이 run을 과거 baseline과
비교할 때 추가 TD smoothing도 달라졌음을 명시한다. 평가 RNG는 별도로 분리해서
평가가 explorer의 데이터 수집 random stream을 소모하지 않게 했다.

## 검증

1. 서로 다른 상수 explorer/evaluator로 데이터 수집 및 평가 routing을 검증한다.
2. 독립적으로 값을 계산할 수 있는 critic으로 target twin-min, terminal mask,
   TD stop-gradient를 검사한다.
3. Actor update가 기존 함수 자체를 그대로 상속함을 확인한다. Explorer의 OT source/KDE는
   EMA로 바꾸지 않는다.
4. 실제 세 MuJoCo 환경에서 48 updates씩 학습하며, 각 update의 새 explorer를 이용한
   EMA 값과 evaluator params가 일치하고 evaluator optimizer step=0인지 확인한다.
5. Full checkpoint 저장/복원 후 states, RNG, 다음 action 및 simulator 결과가 일치하는지 확인한다.
6. W&B verification run의 서버 저장을 확인하고, 본 run의 초기 loss/evaluation을 API로 다시 읽는다.

검증용 runs는 본 seed 결과와 별도 폴더·job_type으로 구분한다.

## 실행과 보관

`heejoon`에 commit/push 후 필요한 tracked source만 snapshot으로 보낸다.
`SOURCE_MANIFEST.json`에 full commit과 파일별 SHA256을 기록한다.
실험 snapshot은 `/home/heejoonorm/OptiQ/legacy_monge/explorer_v1/COMMIT/repo`이고
runs/validation/runtime은 같은 COMMIT 폴더 아래 source 바깥에 둔다.
이 위치는 기존 dildata read-only collector의 범위 안에 있으므로 인증키를 더 복사하지 않는다.
기존 Monge numerical source와 runs는 바꾸지 않는다.

각 환경 worker는 tmux에서 실행하고 GPU 한 장·CPU 두 코어를 사용한다.
Worker가 실패하면 해당 환경의 다음 seed 시작을 중단하고 FAILED.json을 남긴다.
다른 환경 worker에는 의존성을 걸지 않는다. GPU3는 예약만 하고 사용하지 않는다.

매 50K 및 종료/정상 중단 시 explorer·evaluator·critic·optimizer·target params,
replay의 초기화된 구간, RNG, MuJoCo simulator와 callback 상태를 `resume.zip`에 원자적으로 저장한다.
중간 full checkpoint는 최신 하나만 유지하며 dildata가 약 2분마다 회수한다.
같은 source의 재개는 같은 W&B ID와 env_steps를 유지한다. 완료 시 final_models도 저장한다.

중앙 보관 경로:
`dildata:/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/explorer_v1/COMMIT/`.
별도 실험 색인: `/data1/heejoonorm/OptiQ/studies/20260919_v1_heejoon_explorer/`.
W&B 키는 사용자 승인을 받은 별도 0600 파일에 저장하며 source/log/backup에서 제외한다.
