# AntMaze v3 OptiQ, seed 0, 1M 결과

학습과 최종 평가가 완료되었다. 동일한 전체 초기 시뮬레이터 상태에서 각 모드를 100회 평가했다.

| 평가 모드 | 성공 | G1 | G2 | 실패 | 성공 경로 |
|---|---:|---:|---:|---:|---|
| Random z, μ-only | 97/100 | 97 | 0 | 3 | G1/passage-y+4 |
| Random z, conditional sigma 포함 | 99/100 | 99 | 0 | 1 | G1/passage-y+4 |
| Zero z, μ-only | 100/100 | 100 | 0 | 0 | G1/passage-y+4 |

목표 도달은 학습했지만, 이 seed의 최종 평가에서 여러 목표 또는 구분된 성공 경로는 확인되지 않았다. Random z에서도 모든 성공이 G1의 같은 통로로 집중됐다. 궤적 사이의 작은 흔들림을 별도의 성공 경로로 세지 않았다. 이는 한 seed의 결과이며 모든 OptiQ 정책의 특성으로 일반화하지 않는다.

학습 중 최초 성공은 185,390 step, 전체 성공은 4,251개 episode 중 3,700개였다. 총 1,000,000 환경 상호작용, 995,000 learner/RND 업데이트를 수행했다.

v3의 원래 초기화도 고정 상태이므로 natural/fixed 평가를 독립적인 200회로 합산하지 않았다. 위 표와 그림은 fixed 100회만 사용했다. 모든 모드의 전체 초기 시뮬레이터 상태가 동일함을 원본 NPZ에서 확인했다. Random z는 매 행동마다 다시 뽑는다. μ-only는 conditional sigma noise를 사용하지 않으며, policy 모드는 학습된 정책의 sigma를 포함한다. 평가에는 별도 DACER 행동 잡음이나 NovelD 보상을 더하지 않았다.

학습 소스: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`. 저장 검증 모듈 소스: `dd107f29e10adf6a7297c4603aac07a781252292`.

모델·optimizer·RNG·환경 상태·리플레이를 로컬에 보관했다. `runs/v3-optiq-s0/archive-verification.json`이 체크포인트 전체 상태와 실제 리플레이 1,000,000건의 dense reward 검증 통과를 기록한다. 그림에 사용한 원본 NPZ의 SHA256을 해당 검증 기록과 대조했다. 그림별 검증 결과와 plot script 해시는 `fixed-state-verification.json`에 있다.

전체 16개 중 네 번째 완료 결과다. v3 MFPO는 기존 controller가 자동으로 이어서 실행했다. 이 보고는 중간 완료 보고이며 다른 실험의 완료를 의미하지 않는다.

![고정 초기 상태에서 100회씩 평가한 궤적](fixed-state-trajectories.png)
