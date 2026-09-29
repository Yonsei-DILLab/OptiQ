# v4 teacher 온도만 바꾼 geodesic 비교

기존 geodesic T1과 teacher T3/T10을 비교합니다. 알고리즘 구조와 물리 환경은 동일합니다. 거리 보상·할인율·DACER 설정은 같고 teacher 온도만 다릅니다.

| 조건 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 목표별 성공 경로 | 성공률 |
|---|---|---:|---:|---|---|---|---:|
|original-T1|policy|258304|100|{'lower': 73, 'upper': 25, 'uncommitted': 2}|{}|{}|0.0%|
|original-T1|native|258304|100|{'upper': 39, 'lower': 61}|{'upper': 1, 'lower': 1}|{'upper/G1': 1, 'lower/G2': 1}|2.0%|
|T3|policy|258304|100|{'upper': 38, 'lower': 45, 'uncommitted': 17}|{}|{}|0.0%|
|T3|native|258304|100|{'upper': 30, 'lower': 55, 'uncommitted': 15}|{}|{}|0.0%|
|T10|policy|258304|100|{'uncommitted': 98, 'upper': 2}|{}|{}|0.0%|
|T10|native|258304|100|{'uncommitted': 90, 'lower': 7, 'upper': 3}|{}|{}|0.0%|

![같은 예산 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

궤적 그림은 세 조건 모두 저장된 정확히 같은 step을 비교합니다. 표는 최신 결과라 step이 다를 수 있습니다. v4 원래 고정 full state, seed0 하나입니다. 직접 정책은 random z와 conditional sigma를 포함하며 외부 DACER 잡음은 없습니다. mu-only는 별도 보조 결과입니다. 각 reward profile로 보상 합·시작 상태·성공 목표·원시 파일 SHA256을 검증했습니다. 통로 진입만으로 두 성공 경로나 장기간 유지를 주장하지 않습니다.
