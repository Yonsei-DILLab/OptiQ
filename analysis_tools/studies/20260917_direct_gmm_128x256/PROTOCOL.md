# Direct GMM MuJoCo N128×M256 · login4

2026-09-17 사용자 요청: Direct GMM, temperature0.5, Ant → Humanoid → Hopper →
Walker2d → HalfCheetah 순서로 실행. 기존 비교와 같이 환경별 seed0–3, 각1M steps,
총20 runs다. Vast의 N16/N64 작업과 독립적으로 login4 SLURM에서 실행한다.

## 코드와 비교 조건

기존 Direct GMM frozen source372개를 그대로 복사한다. Git source 경로는
`analysis_tools/studies/20260917_nonstationary_q/v5`, source ID는
`fd918514023ae5ce20a91db53f19801549fd9d249e46e963b643d498834f5f28`이다.
새 설정·wrapper·등록·백업 코드를 heejoon에 먼저 commit/push하고,
DEPLOYMENT.json, SLURM 등록 기록, run config에 전체 commit SHA를 남긴다.

| 항목 | 설정 |
|---|---|
| Actor | v5 learned conditional Gaussian, 256×2 GELU |
| Loss | Direct GMM marginal NLL; OT/Sinkhorn 미사용 |
| N / M | 128 / 256; proposals_per_policy_sample=2 |
| Temperature | 고정0.5 |
| Teacher | Conditional Gaussian mixture, IID 후보, sigma floor0.05 |
| Density correction | beta1 |
| Critic | Reward-only twin TD; target min, gamma0.99, Polyak0.005 |
| Update | Actor/critic batch256, UTD1, policy delay1 |
| Optimizer | Adam3e-4, clipping 없음 |
| Warmup / horizon | Uniform5K / 총1M environment steps |
| Evaluation | Dual mean-only(zero-z/sample-z), 5K마다 각각10 episodes |
| Checkpoint | 50K마다 기존 actor/critic/optimizer 저장 |
| W&B | OptiQ/DirectGMM_heejoon, group20260917_DirectGMM_T0.5_N128_M256_login4 |

평가 빈도나 batch를 시간 절약을 위해 임의로 변경하지 않는다. 이전 checkpoint에는
전체 replay/RNG가 없으므로 자동 재시작·자동 requeue를 사용하지 않는다.
실패 seed를 제외하거나 새로운 seed로 바꾸지 않는다.

## 실행 검증과 순서

1. GPU 사용 전20개 Hydra 설정, frozen source hash, 기존 W&B 인증을 확인한다.
2. GPU1개에서 Ant·Humanoid 각각128 steps(32 warmup, 96 updates) 검증을 수행한다.
   실제 batch256/N128/M256 경로의 유한 loss·완료 기록·W&B 온라인 summary를 확인한다.
   두 run은 validation 전용 group이며 본20개 실험에 포함하지 않는다.
3. 검증 성공 후 Ant seed0–3 array를 실행한다. 각 환경의 네 seed가 모두 성공해야
   다음 환경 array가 시작하도록 afterok dependency를 사용한다.

각 seed는 GPU1, CPU2, RAM24GB, wall limit24h를 요청한다. 비선점 big_qos를 사용하고,
기존 검증에서 GPU 노출 문제가 확인됐던 노드는 제외한다. 동시에 최대4 GPU를 쓴다.
하드웨어·패키지 버전은 실제 실행 기록에 남긴다. Vast와의 wall-clock 비교에는
서로 다른 GPU·CPU·소프트웨어 환경이 섞이므로 알고리즘 속도 차이로 단정하지 않는다.

## 보관

- 계산: `login4:/lustre/hobbit9882/OptiQ-DirectGMM-128x256-20260917`.
- 중앙 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260917_direct_gmm_128x256`.
- 인증: login4에 이미 설정된 W&B 인증을 그대로 사용한다. 값은 로그·Git·백업에 넣지 않는다.
- 백업: 별도 read-only SSH key로 이 캠페인 폴더만3분마다 수집한다.
- 완료 run의 ARTIFACTS_SHA256.json을 dildata에서 검증하고 BACKUP_VERIFIED.json을 기록한다.
