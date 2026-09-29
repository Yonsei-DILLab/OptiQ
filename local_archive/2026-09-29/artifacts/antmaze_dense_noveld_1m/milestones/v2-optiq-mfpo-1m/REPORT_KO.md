# AntMaze v2 OptiQ와 MFPO: 1M, seed 0

두 방법 모두 1M 환경 상호작용 학습, 최종 평가, 재개용 체크포인트와 1M replay 로컬 저장 검증을 완료했다. v2 SAC와 MEOW는 아직 학습 중이다.

| 직접 stochastic 정책, 동일한 전체 초기 상태 | 성공/100 | G1 | G2 | 성공 경로 |
|---|---:|---:|---:|---|
| OptiQ | 100 | 100 | 0 | G1/central-corridor |
| MFPO | 100 | 100 | 0 | G1/central-corridor |

두 정책 모두 목적지 도달은 학습했지만 두 목표나 여러 성공 경로로의 분화는 이 seed의 최종 평가에서 관찰되지 않았다. 궤적의 작은 흔들림은 별도 경로로 세지 않았다.

OptiQ의 μ-only와 MFPO의 native Q-best-of-10 평가에서도 각각 100/100회 G1으로 성공했다. 주 그림은 best-of-K 선택 없이 정책에서 행동을 직접 샘플링한다. OptiQ의 직접 샘플링에는 학습된 conditional sigma가 포함되며, 어느 방법에도 외부 DACER 잡음이나 NovelD 보상을 평가에 추가하지 않았다.

OptiQ는 학습 중 첫 성공 26,829 step, MFPO는 64,623 step이었다. MFPO는 총 23,922개 training episode 중 23,747개가 성공했다. 두 방법의 1M 환경 예산은 같지만 warmup 차이로 learner 업데이트는 OptiQ 995k, MFPO 990k다. 그 밖에도 원래 모델·optimizer·entropy backup 설정을 유지했으므로 정책 구조만의 단일 요인 ablation이 아니다.

v2의 natural 초기화도 고정 상태이므로 natural/fixed 평가를 독립적인 200회로 합산하지 않았다. 위 표와 그림은 fixed 100회씩만 사용했다. 전체 초기 시뮬레이터 상태가 두 방법과 모든 rollout에서 동일함을 원본 NPZ로 확인했다.

학습 소스 `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`, 저장 검증 소스 `dd107f29e10adf6a7297c4603aac07a781252292`. `runs/v2-{optiq,mfpo}-s0/archive-verification.json`에 파일 SHA, 전체 checkpoint 상태 및 실제 replay 1M건의 dense reward 검증이 기록되어 있다. 그림 입력 NPZ의 SHA를 해당 검증 기록과 대조했으며 `fixed-state-verification.json`에 집계를 저장했다.

![같은 초기 상태에서 두 정책의 직접 stochastic rollout](fixed-state-comparison.png)
