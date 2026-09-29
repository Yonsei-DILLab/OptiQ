# 저장 로그·궤적 진단

학습 소스: `f953d28456d3800860dddb9b9cb91b6bd520ae00`. 읽기 분석만 수행. 기존 학습·평가·설정을 변경하거나 새 평가를 실행하지 않았다. 4090은 사용하지 않았다.

## 최초 성공 기록 재확인

v3는 완료된 `training-successes.json`으로 최초 **178,960 total transitions (global170,768), env15, goal2**를 확인했다. 두 번째 학습 성공은466,177이다. v4는 아직 최종 성공 목록이 저장되지 않았으므로4,096 transition 로그구간을 유지한다. 전체progress레코드를 처음부터 재검사했으며 두환경모두 successes필드누락0, JSON파싱실패0,누적카운터감소0이다. 평가성공은 별도환경의 결과이며학습성공카운터를증가시키지않는다.

## 가장 중요한 관측

**v4는 학습 중 goal 성공 이전에 이미 상단 경로로 집중됐다.** 따라서 “성공에 따른 terminal 이득이 발생한 순간에 경로가 사라졌다”는 설명은 v4에서 성립하지 않는다. 성공 이전의 연속적인 progress reward 또는 actor/critic 학습 편향은 여전히 가능하다.

## v3

첫 학습 성공은 total transitions (176,128, 180,224] 사이. warmup 8,192를 제외한 global steps로는 (167,936, 172,032] 사이다. 로그가 4,096 transition마다 기록되므로 정확한 1개 transition은 완료 시 training-successes.json에서 확인 가능하다.

| 평가 total step | 경로 횟수 | 평가 성공 |
|---:|---|---:|
| 250,112 | {'left': 12, 'right': 23, 'uncommitted': 5} | 0/40 |
| 500,224 | {'right': 39, 'uncommitted': 1} | 16/40 |
| 750,080 | {'right': 39, 'uncommitted': 1} | 37/40 |
| 1,000,192 | {'right': 40} | 40/40 |

250k에서 실제 trajectory로 재계산한 gamma=.99 return (평균±sample SD):

- left: n=12, 할인 return=546.01 ± 49.69, 무할인합계=1242.53, 최종 goal 거리=4.55m.
- right: n=23, 할인 return=550.54 ± 65.33, 무할인합계=1321.23, 최종 goal 거리=3.76m.
- uncommitted: n=5, 할인 return=461.56 ± 50.94, 무할인합계=930.12, 최종 goal 거리=7.67m.

| learner 로그 구간 | 평균 raw sigma | teacher ESS / 64 | 최대 teacher weight | 평균 source Q 표준편차 |
|---|---:|---:|---:|---:|
| [160000] | 0.36691 | 18.21 | 0.192 | 1.976 |
| [320000, 480000] | 0.36667 | 18.84 | 0.185 | 1.882 |
| [640000] | 0.36721 | 20.15 | 0.176 | 1.746 |

## v4

첫 학습 성공은 total transitions (831,488, 835,584] 사이. warmup 8,192를 제외한 global steps로는 (823,296, 827,392] 사이다. 로그가 4,096 transition마다 기록되므로 정확한 1개 transition은 완료 시 training-successes.json에서 확인 가능하다.

| 평가 total step | 경로 횟수 | 평가 성공 |
|---:|---|---:|
| 250,112 | {'upper': 28, 'uncommitted': 3, 'lower': 9} | 0/40 |
| 500,224 | {'upper': 40} | 0/40 |
| 750,080 | {'upper': 40} | 14/40 |
| 1,000,192 | {'upper': 40} | 27/40 |

250k에서 실제 trajectory로 재계산한 gamma=.99 return (평균±sample SD):

- upper: n=28, 할인 return=192.69 ± 51.61, 무할인합계=815.32, 최종 goal 거리=8.34m.
- uncommitted: n=3, 할인 return=132.17 ± 13.53, 무할인합계=256.39, 최종 goal 거리=13.93m.
- lower: n=9, 할인 return=135.18 ± 30.00, 무할인합계=619.07, 최종 goal 거리=10.30m.

| learner 로그 구간 | 평균 raw sigma | teacher ESS / 64 | 최대 teacher weight | 평균 source Q 표준편차 |
|---|---:|---:|---:|---:|
| [160000] | 0.36689 | 21.73 | 0.164 | 1.246 |
| [320000, 480000] | 0.36668 | 22.82 | 0.157 | 1.425 |
| [640000] | 0.36370 | 22.85 | 0.156 | 1.652 |

## 해석 한계

- teacher ESS와 sigma는 replay minibatch를 평균낸 전역 로그다. 분기 상태에 국한된 teacher/actor 붕괴 여부를 판정하지 못한다. 그러나 전역 sigma가 0으로 줄거나 전역 teacher가 단일 후보로 집중된 흔적은 없다.
- learner CSV는 160k transition 간격으로 각각 8개 연속 update를 기록한다. 구간 평균은 학습 전체의 시간평균이 아니다.
- 250k 경로별 return은 사후에 경로로 분류한 rollout 표본이다. 동일 분기 상태에서 행동만 바꾸는 반사실적 비교가 아니며, critic Q 자체도 아니다.
- v4 상단은 초기부터 더 깊이 진행했고 실현 return도 더 높았다. 이것은 지도의 본질적 상하 비대칭을 증명하지 않으며, 해당 시점 정책의 제어 능력 차이일 수 있다.
- 학습 성공 카운트는 learner에 들어가는 training 환경의 성공 기록이고 evaluation의 성공은 이 카운터에 추가되지 않는다. 평가 환경이 별도로 생성됨을 run.py에서 확인했다.
- 40회 중 한 경로가 0회라는 것은 그 확률이 엄밀히 0이라는 뜻이 아니다. 단일 seed 결과이며 확률적인 미관측 경로는 남아 있을 수 있다.
