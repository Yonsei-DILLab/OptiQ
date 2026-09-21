# Direct GMM MuJoCo: N64 × M256, temperature 0.5

2026-09-17 사용자 요청으로 추가 예약한다. 기존 N16×M64 캠페인의 모든 환경이
성공적으로 완료된 뒤 Ant → Humanoid → Hopper → Walker2d → HalfCheetah를 실행한다.
환경별 seed 0–3, 각 1M steps이며 네 GPU에서 seed를 하나씩 병렬 실행한다.
한 환경의 네 seed가 모두 끝나야 다음 환경으로 넘어간다. 총 20 runs.

## 통제한 설정

- Direct GMM marginal NLL, v5 learned conditional Gaussian actor.
- N=64, M=256, proposals_per_policy_sample=4, temperature=0.5.
- 기존 N16×M64의 frozen code 372개를 그대로 재사용하고 실행 전 hash를 검사한다.
- 소스 Git 경로: `analysis_tools/studies/20260917_nonstationary_q/v5`.
- Source ID: `fd918514023ae5ce20a91db53f19801549fd9d249e46e963b643d498834f5f28`.
- Actor/critic 256×2 GELU, learned sigma, Adam 3e-4, clipping 없음.
- Actor/critic batch256, UTD1, policy delay1, warmup5K.
- Reward-only twin TD, gamma0.99, Polyak0.005.
- 평가: 기존 dual mean-only 2종류, 5K마다 각10 episodes; checkpoint50K.
- 변경되는 실험 변수는 N 및 그에 따른 M뿐이다. 온도·평가 빈도·학습 길이는 유지한다.

W&B: `OptiQ/DirectGMM_heejoon`, group `20260917_DirectGMM_T0.5_N64_M256`.
Online 기록을 유지하고 각 run config에 사전 commit SHA를 기록한다.
사전 검증은 전체20개 Hydra config와 기존 W&B project 접근을 확인한다.
예약 시점에는 새 training run을 만들지 않으며, 실제 새 metrics는 실행 후 생성된다.

## 실행과 보관

- Vast: 기존 사용자 지정 `45.143.122.5:35221`, instance51277568, RTX4090×4.
- 예약·로그·결과: `/workspace/DirectGMM/campaigns/20260917_N64_M256/`.
- 공통 frozen code: `/workspace/DirectGMM/repo` (수정하지 않음).
- tmux: `direct-gmm-64x256-queue`.
- dildata 결과: `/data1/heejoonorm/OptiQ/runs/vast-51277568/DirectGMM_N64_M256`.
- dildata 예약 자료: `/data1/heejoonorm/OptiQ/studies/20260917_direct_gmm_64x256`.
- dildata collector tmux: `direct-gmm-51277568-n64-sync`, 2분 간격.

기존 controller의 성공 표시와 GPU 실행 lock 해제를 모두 기다린다. 기존 queue와
학습 프로세스는 수정하지 않는다. 선행 작업 실패 시 새 캠페인은 대기한다.
새 캠페인에서 실패가 발생하면 같은 환경의 다른 seed는 마치고 이후 환경은 보류한다.
이전 checkpoint는 replay/RNG 전체를 포함하지 않으므로 중단된 run을 자동 재시작하지 않는다.
완료 결과는 SHA256 manifest를 만들고 dildata에서 재검증한다. 자격증명은 별도 경로의
기존 파일을 사용하며 Git·캠페인 디렉터리·백업에 넣지 않는다.

`DEPLOYMENT.json`은 이 파일들을 commit·push한 뒤 생성하며, commit SHA와 배포 파일
hash를 기록한다. `PREFLIGHT.json`과 `queue.json`은 배포·예약 후 생성되는 운영 기록이다.
