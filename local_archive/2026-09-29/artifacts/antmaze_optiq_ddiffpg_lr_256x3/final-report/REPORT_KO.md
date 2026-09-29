# OptiQ AntMaze 완료 보고

- 학습 source: `cac1365fd9878d46874bbc09a840816ba1461499`
- 256×3 actor/critic, actor LR 3e-4, critic LR 5e-4, T=0.01, sparse + NovelD 0.01, seed0.
- 256개 병렬 환경, batch4096, 256 transition당 8 updates.
- 전체 완료: 2026-09-23 21:31 KST. 4개 모두 정상 종료.
- 아래 평가는 최종 저장 checkpoint에서 서로 다른 랜덤 시작 100회.

| 환경 | 완료 interactions | 직접 정책 성공 | mu-only 성공 | 학습 중 성공 | 학습 시간 |
|---|---:|---:|---:|---:|---:|
| v1 | 3,008,256 | 0/100 | 0/100 | 0 | 52.7분 |
| v2 | 3,008,256 | 98/100 | 88/100 | 8,931 | 70.3분 |
| v3 | 4,008,448 | 0/100 | 0/100 | 0 | 76.9분 |
| v4 | 5,008,384 | 0/100 | 0/100 | 0 | 91.3분 |

직접 정책은 random z + conditional sigma, mu-only는 random z를 유지하고 conditional sigma만 제거한다. 평가에서 외부 DACER 잡음이나 NovelD 보상을 더하지 않는다.

v2 학습 중 첫 성공은 2,033,975 interactions이며 8,931회 모두 G2였다. 최종 직접 정책은 왼쪽 통로 진입 0회, 오른쪽 진입/성공 98회, 어느 쪽에도 진입 못한 실패 2회였다. G1 성공은 0회다. 고정 초기상태 평가도 G2로만 100/100 성공했다. 성공률이 높아졌으나 여러 목표/경로로 성공하는 정책은 아니다.

v1 zero_z 랜덤 시작 보조평가에서만 1/100 성공했다. v1·v3·v4의 학습 중 성공은 모두 0회였다. 네 방법 비교가 아닌 OptiQ의 단일 seed 실험이다.

W&B OptiQ/antmaze의 네 run은 모두 finished를 확인했다. W&B 기본 eval 카드의 v2 95%/85%는 마지막 정기평가 40회 결과이고, 위 98%/88%는 학습 종료 후 별도로 저장한 100회 평가다. 현재 학습 코드는 최종 100회 결과를 기본 eval summary 카드에 덮어쓰지 않는다.

24개 원시 NPZ의 2,400 episodes에서 길이·유한성·도달 거리·goal 라벨·평균 return·초기상태 다양성·저장 summary 일치를 검사했다. JSON과 원시 평가는 본 디렉터리에 보관했다. 최종 full checkpoint는 서버에 보존되며 저장 시 readback 및 sparse replay 보상 검증을 통과했다. 이번 보고에서는 full checkpoint 전체를 로컬로 복사하지 않았다.
