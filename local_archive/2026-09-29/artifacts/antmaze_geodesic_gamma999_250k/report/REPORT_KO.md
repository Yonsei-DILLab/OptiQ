# Geodesic 거리만 바꾼 gamma .999 비교

동일 seed0, gamma .999, T1, DACER 목표 +.7/차원·500update 간격입니다. 알고리즘 변경 없이 기존 거리 보상만 비교합니다.

| 환경 | 거리 | 평가 | step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |
|---|---|---|---:|---:|---|---|---:|
|v1|euclidean|native|258304|100|{'uncommitted': 100}|{}|0.0%|
|v1|euclidean|policy|258304|100|{'uncommitted': 100}|{}|0.0%|
|v1|geodesic|native|258304|100|{'lower': 41, 'upper': 51, 'uncommitted': 8}|{'upper': 13, 'lower': 6}|19.0%|
|v1|geodesic|policy|258304|100|{'lower': 39, 'uncommitted': 9, 'upper': 52}|{'lower': 9, 'upper': 17}|26.0%|
|v3|euclidean|native|258304|100|{'left': 54, 'right': 20, 'uncommitted': 26}|{}|0.0%|
|v3|euclidean|policy|258304|100|{'left': 54, 'right': 33, 'uncommitted': 13}|{}|0.0%|
|v3|geodesic|native|258304|100|{'right': 90, 'uncommitted': 10}|{'right': 12}|12.0%|
|v3|geodesic|policy|258304|100|{'right': 100}|{'right': 25}|25.0%|
|v4|geodesic|native|258304|100|{'upper': 39, 'lower': 61}|{'upper': 1, 'lower': 1}|2.0%|
|v4|geodesic|policy|258304|100|{'lower': 73, 'upper': 25, 'uncommitted': 2}|{}|0.0%|

![동일 step 직접 정책](matched_trajectories_policy.png)
![mu-only 보조](matched_trajectories_native.png)

각 그림 행은 정확히 동일한 step만 비교합니다. 위 표는 가장 최근 저장 평가이므로 step이 다르면 직접 성능 비교를 하지 않습니다. 반대 통로 방문과 그 경로로 실제 성공하는 것을 구분합니다. v1은 학습과 같은 랜덤 시작이며, v3/v4는 원래 고정 full state입니다. 학습 source·원시 궤적 SHA256·실제 거리 보상 telescope를 검증하고 학습 자료는 변경하지 않았습니다.
