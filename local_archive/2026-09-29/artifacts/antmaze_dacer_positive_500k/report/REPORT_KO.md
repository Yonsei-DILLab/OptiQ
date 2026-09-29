# DACER 양수 entropy 500k 전체 결과

16개 모두 508416 total transitions(500224 post-warmup)와 최종 100회 평가·체크포인트 검증을 완료했습니다. 지속적인 다중 성공 경로는 아직 입증되지 않았습니다. 각 패널은 서로 다른 단일 seed0 정책입니다.

| 환경 | H/차원 | total step | 평가수 | 첫 통로 진입 | 성공 통로 | 성공률 |
|---|---:|---:|---:|---|---|---:|
| v1 | +0.1 | 508416 | 100 | {'uncommitted': 87, 'upper': 12, 'lower': 1} | {} | 0.0% |
| v1 | +0.5 | 508416 | 100 | {'uncommitted': 74, 'upper': 22, 'lower': 4} | {} | 0.0% |
| v1 | +0.7 | 508416 | 100 | {'uncommitted': 70, 'upper': 25, 'lower': 5} | {} | 0.0% |
| v1 | +0.9 | 508416 | 100 | {'uncommitted': 64, 'upper': 30, 'lower': 6} | {} | 0.0% |
| v2 | +0.1 | 508416 | 100 | {'right': 100} | {'right': 100} | 100.0% |
| v2 | +0.5 | 508416 | 100 | {'right': 100} | {'right': 100} | 100.0% |
| v2 | +0.7 | 508416 | 100 | {'right': 100} | {'right': 100} | 100.0% |
| v2 | +0.9 | 508416 | 100 | {'right': 100} | {'right': 100} | 100.0% |
| v3 | +0.1 | 508416 | 100 | {'uncommitted': 79, 'right': 21} | {'right': 1} | 1.0% |
| v3 | +0.5 | 508416 | 100 | {'left': 97, 'uncommitted': 3} | {'left': 48} | 48.0% |
| v3 | +0.7 | 508416 | 100 | {'left': 100} | {'left': 78} | 78.0% |
| v3 | +0.9 | 508416 | 100 | {'left': 100} | {'left': 19} | 19.0% |
| v4 | +0.1 | 508416 | 100 | {'lower': 100} | {} | 0.0% |
| v4 | +0.5 | 508416 | 100 | {'upper': 99, 'lower': 1} | {} | 0.0% |
| v4 | +0.7 | 508416 | 100 | {'upper': 96, 'uncommitted': 2, 'lower': 2} | {} | 0.0% |
| v4 | +0.9 | 508416 | 100 | {'upper': 97, 'uncommitted': 3} | {} | 0.0% |

통로 진입과 goal 도달은 다릅니다. Random latent+conditional sigma 직접정책을 사용하고 외부 DACER 잡음은 평가에서 제외합니다. v1은 원래 랜덤 시작, v2-v4는 원래 고정 full state입니다.

![최근 궤적](latest_trajectories.png)
![경로 유지와 성공](route_retention.png)
![조절 반응](entropy_response.png)

Reward100*(d_current-d_next), bonus0·step penalty0·NovelD OFF, T1. 원시 NPZ episode/성공/goal endpoint/누적 reward를 검증했고 각 SHA256과 보고 코드 해시는 results.json에 보관했습니다. 학습 source2564b59는 그대로입니다.

후속 gamma.999/T3 및 조합의250k screen8개는 학습 sourceeee04de, controller7278a0e이며 핵심 알고리즘7파일은2564b59와 byte 단위 동일합니다. 완료된 v3 gamma.999/T1은 최종100회에서 left54/right33, success0이었고 mu-only에서도 left54/right20, success0입니다. 이 조건만 같은 seed0로 새로1M 학습하는 후속을 source d266263으로 등록했습니다. 이는 독립 seed 또는 checkpoint 재개가 아닙니다. 목표를 달성했다고 판정하지 않았습니다.
