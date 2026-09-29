# AntMaze annealing 최종 결과

모두 dense reward + NovelD OFF, training seed 0. 성공률은 최종 100회 random-start direct-policy 평가다. OptiQ는 random latent + conditional sigma를 포함하며 외부 DACER 잡음은 없다.

| 설정 | v1 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|
| T=0.01 | 98% | 92% | 0% | 86% |
| T=1 | 100% | 94% | 96% | 84% |
| 10->1 | 100% | 98% | 100% | 83% |
| 10->0.25 | 99% | 99% | 100% | 78% |
| 10->0.5 | 100% | 89% | 100% | 65% |
| SAC | 0% | 100% | 3% | 0% |
| DIPO | 98% | 88% | 0% | 84% |
| MFPO | 23% | 83% | 83% | 62% |

Warmup 8,192 transition 이후 1M 동안 선형 감소하고 최종 온도를 유지했다. 전체 예산은 기존 v1/v2 3M, v3 4M, v4 5M (+warmup 및 vector step 반올림).

## Annealing 경로 비율

| 설정 | 환경 | random-start 성공 경로 | 동일 full-state 성공 경로 |
|---|---|---|---|
| 10->1 | v1 | {"G1/lower": 38, "G1/upper": 62} | {"G1/lower": 99, "failure": 1} |
| 10->1 | v2 | {"failure": 2, "goal(8,0)/central-corridor": 98} | {"failure": 34, "goal(8,0)/central-corridor": 66} |
| 10->1 | v3 | {"G2/passage-y-8": 100} | {"G2/passage-y-8": 100} |
| 10->1 | v4 | {"G1/upper-entry/upper-outer": 83, "failure": 17} | {"G1/upper-entry/upper-outer": 91, "failure": 9} |
| 10->0.25 | v1 | {"G1/lower": 8, "G1/upper": 91, "failure": 1} | {"G1/lower": 1, "G1/upper": 99} |
| 10->0.25 | v2 | {"failure": 1, "goal(8,0)/central-corridor": 99} | {"goal(8,0)/central-corridor": 100} |
| 10->0.25 | v3 | {"G2/passage-y-8": 100} | {"G2/passage-y-8": 100} |
| 10->0.25 | v4 | {"G1/upper-entry/upper-outer": 78, "failure": 22} | {"failure": 100} |
| 10->0.5 | v1 | {"G1/lower": 20, "G1/upper": 80} | {"G1/lower": 99, "failure": 1} |
| 10->0.5 | v2 | {"failure": 11, "goal(8,0)/central-corridor": 89} | {"failure": 47, "goal(8,0)/central-corridor": 53} |
| 10->0.5 | v3 | {"G2/passage-y-8": 100} | {"G2/passage-y-8": 99, "failure": 1} |
| 10->0.5 | v4 | {"G1/upper-entry/upper-outer": 65, "failure": 35} | {"failure": 100} |

성공 경로는 기하학적 통로 교차로 분류했다. 랜덤 시작점에 따른 경로 차이를 동일 상태에서의 정책 multimodality로 해석하지 않는다. 단일 학습 seed이며, 100회 평가에 관찰되지 않은 드문 경로의 부재를 증명하지 않는다.

원시 궤적의 유한성·padding, 목표 반경 진입, dense return 합, 초기 상태, SHA256를 검증했다. 설정·평가 NPZ·검증 sidecar를 로컬에 보관했고 전체 모델/replay checkpoint는 서버에 남아 있다.

학습 source 및 각 결과 출처는 analysis.json에 기록했다. W&B 199 서버의 offline 자료는 네트워크 복구 후 동기화 대기 중이다.
