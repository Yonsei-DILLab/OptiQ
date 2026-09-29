# v3 시작점 기준 거리 정규화 비교

기존 geodesic T1과 정규화 T1/T3를 비교합니다. 알고리즘 구조와 물리 환경은 동일합니다. 정규화는 목표별 고정 거리 가중치와 기존 성공 반경까지의 거리 두 가지를 함께 바꿉니다.

| 조건 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 목표별 성공 경로 | 성공률 |
|---|---|---:|---:|---|---|---|---:|
|original-T1|policy|258304|100|{'right': 100}|{'right': 25}|{'right/G2': 25}|25.0%|
|original-T1|native|258304|100|{'right': 90, 'uncommitted': 10}|{'right': 12}|{'right/G2': 12}|12.0%|
|normalized-T1|policy|258304|100|{'right': 87, 'uncommitted': 13}|{}|{}|0.0%|
|normalized-T1|native|258304|100|{'uncommitted': 26, 'right': 74}|{'right': 1}|{'right/G2': 1}|1.0%|
|normalized-T3|policy|258304|100|{'right': 84, 'left': 16}|{'right': 4}|{'right/G2': 4}|4.0%|
|normalized-T3|native|258304|100|{'right': 92, 'left': 8}|{'right': 38}|{'right/G2': 38}|38.0%|

![같은 예산 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

궤적 그림은 세 조건 모두 저장된 정확히 같은 step을 비교합니다. 표는 최신 결과라 step이 다를 수 있습니다. v3 원래 고정 full state, seed0 하나입니다. 직접 정책은 random z와 conditional sigma를 포함하며 외부 DACER 잡음은 없습니다. mu-only는 별도 보조 결과입니다. 각 reward profile로 보상 합·시작 상태·성공 목표·원시 파일 SHA256을 검증했습니다. 통로 진입만으로 두 성공 경로나 장기간 유지를 주장하지 않습니다.
