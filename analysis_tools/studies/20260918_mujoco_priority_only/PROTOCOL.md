# MuJoCo environment order means priority, not dependency

2026-09-18 사용자 정정: 환경 순서는 우선순위였으며, GPU를 최대한 활용해
서로 독립적인 작업들을 동시에 실행한다. 이 지시가 이전 순차 실행 protocol을 대체한다.

login4의 N128×M256, temperature0.5/0.25, seed0–3, 각1M steps를 유지한다.
중단된 Ant T0.5/seed3와 T0.25/seed2는 이전 지시에 따라 그대로 보류한다.
현재 실행 중인 Humanoid8개는 유지한다. Hopper8개, Walker2d8개, HalfCheetah8개
총24개 pending run의 dependency를 모두 해제하여 총32개가 독립적으로 실행 가능하게 한다.
실제 동시 GPU 수는 SLURM의 가용 자원과 사용자 제한에 따라 결정된다.

| Environment | Nice (낮을수록 우선) | T0.5 array | T0.25 array |
|---|---:|---|---|
| Humanoid | 0 (기존 값, 실행 중) | 2277774 | 2277873 |
| Hopper | 10 | 2277775 | 2277874 |
| Walker2d | 20 | 2277776 | 2277875 |
| HalfCheetah | 30 | 2277777 | 2277876 |

Nice는 같은 환경 cohort 사이의 scheduling preference이며 다른 환경의 완료를
기다리지 않는다. SLURM 자원 fit/backfill과 클러스터 priority 때문에 시작 순서는
엄격하게 보장하지 않으며, 이를 위해 GPU를 의도적으로 비우지 않는다.
각 seed는 기존 GPU1/CPU2/RAM24GB를 유지하며, 별도 전체 동시 실행 제한을 추가하지 않는다.
각 array의 `%4`는 해당 환경 seed가4개이므로 총32개 동시 실행을 제한하지 않는다.

기존 수치 코드·하이퍼파라미터·W&B group·job IDs·백업은 유지한다.
원래 source commit은 T0.5 `ec9bb32b424a37147ea6f3dacf08e9315f28a622`,
T0.25 `23f226e0da65cf493afe27b9087a364d534a81cd`다.
운영 변경 commit과 변경 전후 scheduler snapshot은 각 캠페인의
`PRIORITY_ONLY_20260918.json`에 따로 저장한다. Ant를 완료 처리하지 않는다.

Vast에는 이번 login4 스케줄러 수정이 적용되지 않는다. Vast의4개 GPU는 이미
실행 중인 작업에 사용 중이며, 추가 유료 인스턴스를 대여하지 않는다.
