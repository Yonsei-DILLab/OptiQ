# AntMaze v2 OptiQ, seed 0, 1M 결과

학습과 최종 평가를 완료했으며 재개용 체크포인트와 실제 리플레이 1,000,000건의 로컬 저장 검증을 통과했다.

| 동일한 전체 초기 상태에서 100회 평가 | 성공 | G1 | G2 | 성공 경로 |
|---|---:|---:|---:|---|
| Random z, μ-only | 100/100 | 100 | 0 | G1/central-corridor |
| Random z, conditional sigma 포함 | 100/100 | 100 | 0 | G1/central-corridor |
| Zero z, μ-only | 100/100 | 100 | 0 | G1/central-corridor |

이 seed의 최종 평가에서는 목표 도달에 성공하지만 모든 성공이 G1의 중앙 통로에 집중됐다. 두 번째 목표나 구분되는 여러 성공 경로는 확인되지 않았다. 궤적의 작은 흔들림은 별도 경로로 세지 않았다. 다른 알고리즘의 v2 최종 결과는 아직 완료되지 않아 동일 예산 비교 결론을 내리지 않는다.

최초 학습 성공은 26,829 step이었다. 전체 학습에서 20,221개 episode 중 20,061개가 성공했다. 총 1M 환경 상호작용과 995k learner/RND 업데이트를 수행했다.

v2는 원래 초기화도 고정 상태이므로 natural/fixed 평가를 독립적인 200회로 합산하지 않았다. 표와 그림은 fixed 100회만 사용했다. 모든 모드의 전체 초기 시뮬레이터 상태가 동일함을 원본 NPZ에서 확인했다. Random z는 매 행동마다 새로 샘플링한다. μ-only에서는 conditional sigma noise를 제거하며, policy에서는 학습된 sigma를 포함한다. 평가에 외부 DACER 잡음이나 NovelD 보상을 추가하지 않는다.

학습 소스는 `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`, 저장 검증 모듈은 `dd107f29e10adf6a7297c4603aac07a781252292`이다. 모델·optimizer·RNG·환경 상태·리플레이가 로컬 `runs/v2-optiq-s0`에 보관되어 있다. `archive-verification.json`에 전체 상태와 실제 replay dense reward 검증이 기록되어 있으며, 그림에 사용한 원본 NPZ의 SHA256을 그 기록과 대조했다. 그림별 검증과 plot script 해시는 `fixed-state-verification.json`에 있다.

전체 16개 중 다섯 번째 완료 결과다. 기존 controller가 같은 GPU에 v4 OptiQ를 이어서 실행했다. 나머지 실험은 계속 진행 중이다.

![고정 초기 상태에서 100회씩 평가한 궤적](fixed-state-trajectories.png)
