# v3 gamma .999·T3 경로 유지 추적

완료한 T1 대조군, T3 250k 탐색 실험, 별도의 fresh T3 1M 실험을 비교합니다. 모두 seed0이며 짧은 실험과 긴 실험을 독립 seed로 세지 않습니다.

| 실험 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 성공률 |
|---|---|---:|---:|---|---|---:|
|T1_1M|policy|1008384|100|{'left': 99, 'uncommitted': 1}|{'left': 74}|74.0%|
|T1_1M|native|1008384|100|{'left': 98, 'uncommitted': 2}|{'left': 71}|71.0%|
|T3_250k|policy|258304|100|{'left': 20, 'right': 80}|{'right': 33}|33.0%|
|T3_250k|native|258304|100|{'left': 17, 'right': 79, 'uncommitted': 4}|{'right': 49}|49.0%|
|T3_1M|policy|550144|40|{'right': 39, 'uncommitted': 1}|{'right': 26}|65.0%|
|T3_1M|native|550144|40|{'right': 40}|{'right': 34}|85.0%|

![현재 궤적](latest_trajectories.png)
![학습 추이](learning_curves_policy.png)

원시 궤적의 동일 full state·성공 위치·보상·SHA256 및 완료한 실험의 최종 체크포인트 증명을 검증합니다. 학습 알고리즘은 바꾸지 않았으며 통로 진입과 그 통로를 통한 목표 성공을 분리해 기록합니다.

T3 후속은450k·500k 평가에서 양쪽 모드 모두 반대 경로를 잃어 조기 중단했습니다. 마지막 기록된 학습 step은598016입니다. 계획한1M을 완료한 실험이 아니며 최종100회 평가·전체 replay 체크포인트는 없습니다. 중간 체크포인트, 원시 평가, 로그와 frozen source는 보존했습니다.
