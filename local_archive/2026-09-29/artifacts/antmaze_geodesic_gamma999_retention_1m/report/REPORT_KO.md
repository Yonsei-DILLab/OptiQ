# Geodesic v1/v4 양방향 성공 유지 검증

250k 탐색 실험과 동일한 설정으로 새로 시작한1M 실험을 비교합니다. 모두 seed0이고 독립 학습 seed 또는 체크포인트 재개가 아닙니다.
v4는 반대쪽 성공 경로 소실로 조기 중단했습니다. 마지막 학습 로그는 626688total step이며, 1M 완료·최종100회 평가·최종 full checkpoint가 아닙니다. v1의 완료 결과는 보존했습니다.


| 환경 | 예산 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |
|---|---|---|---:|---:|---|---|---:|
|v4|short|policy|258304|100|{'lower': 73, 'upper': 25, 'uncommitted': 2}|{}|0.0%|
|v4|short|native|258304|100|{'upper': 39, 'lower': 61}|{'upper': 1, 'lower': 1}|2.0%|
|v1|short|policy|258304|100|{'lower': 39, 'uncommitted': 9, 'upper': 52}|{'lower': 9, 'upper': 17}|26.0%|
|v1|short|native|258304|100|{'lower': 41, 'upper': 51, 'uncommitted': 8}|{'upper': 13, 'lower': 6}|19.0%|
|v4|long|policy|600064|40|{'lower': 39, 'uncommitted': 1}|{'lower': 20}|50.0%|
|v4|long|native|600064|40|{'lower': 40}|{'lower': 13}|32.5%|
|v1|long|policy|1008384|100|{'lower': 46, 'upper': 52, 'uncommitted': 2}|{'lower': 46, 'upper': 52}|98.0%|
|v1|long|native|1008384|100|{'lower': 53, 'upper': 46, 'uncommitted': 1}|{'lower': 53, 'upper': 44}|97.0%|

![현재 궤적](latest_trajectories.png)
![직접 정책 추이](learning_curves_policy.png)

v1은 원래 랜덤 시작이므로 양방향 성공만으로 동일 상태의 다중 경로를 입증하지 않습니다. v4는 원래 고정 full state입니다. 그림은 환경마다 두 평가 모드의 최신 공통 checkpoint를 사용합니다. 표는 모드별 최신 저장 평가라 step이 다를 수 있습니다. 직접 정책과 mu-only를 합치지 않고, 실패도 모두 분모에 포함합니다. 원시 보상·goal·시작 상태·SHA256 및 완료한 실험의 체크포인트 증명을 검증했습니다.
