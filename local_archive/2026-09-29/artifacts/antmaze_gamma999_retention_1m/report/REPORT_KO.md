# v3 gamma .999 장기 유지 검증

짧은 실험에서 남은 양방향 진입이 성공까지 이어지는지 확인하는 별도 fresh seed0 실험입니다. 체크포인트 재개 또는 독립 seed로 해석하지 않습니다.

| 실험 | 평가 | total step | 횟수 | 첫 통로 | 성공 통로 | 성공률 |
|---|---|---:|---:|---|---|---:|
| control | native | 508416 | 100 | {'left': 100} | {'left': 68} | 68.0% |
| control | policy | 508416 | 100 | {'left': 100} | {'left': 78} | 78.0% |
| long | native | 1008384 | 100 | {'left': 98, 'uncommitted': 2} | {'left': 71} | 71.0% |
| long | policy | 1008384 | 100 | {'left': 99, 'uncommitted': 1} | {'left': 74} | 74.0% |
| short | native | 258304 | 100 | {'left': 54, 'right': 20, 'uncommitted': 26} | {} | 0.0% |
| short | policy | 258304 | 100 | {'left': 54, 'right': 33, 'uncommitted': 13} | {} | 0.0% |

![현재 궤적](latest_trajectories.png)
![직접 정책 추이](learning_curves_policy.png)
![mu-only 보조 추이](learning_curves_native.png)

동일한 원래 시작 full state에서 평가합니다. 외부 DACER 행동잡음은 제외합니다. 모든 원시 궤적은 시작상태·성공/goal endpoint·reward telescope·SHA256을 검증합니다. 500k/750k/1M에서 양쪽 성공 경로가 유지되는지를 보고, 유망할 경우 추가 독립 평가·학습 seed 검증이 필요합니다.
