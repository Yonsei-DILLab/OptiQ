# 기존 선형 보간을 사용하는 온도 상승 실험

고정 T1과1→3 스케줄을 같은 보상·할인율에서 비교합니다. v1은 원래 랜덤 시작, v4는 원래 고정 full state입니다. v1의 양방향 성공만으로 동일 상태의 조건부 다중 경로를 주장하지 않습니다.

|환경|조건|mode|total step|학습 T|평가 수|진입|성공 통로|
|---|---|---|---:|---:|---:|---|---|
|v4|fixed-T1|policy|600064|1.000|40|{'lower': 39, 'uncommitted': 1}|{'lower': 20}|
|v4|fixed-T1|native|600064|1.000|40|{'lower': 40}|{'lower': 13}|
|v1|fixed-T1|policy|1008384|1.000|100|{'lower': 46, 'upper': 52, 'uncommitted': 2}|{'lower': 46, 'upper': 52}|
|v1|fixed-T1|native|1008384|1.000|100|{'lower': 53, 'upper': 46, 'uncommitted': 1}|{'lower': 53, 'upper': 44}|
|v4|linear-T1to3|policy|900096|2.784|40|{'lower': 40}|{}|
|v4|linear-T1to3|native|900096|2.784|40|{'lower': 40}|{}|
|v1|linear-T1to3|policy|1008384|3.000|100|{'lower': 40, 'upper': 60}|{'lower': 38, 'upper': 57}|
|v1|linear-T1to3|native|1008384|3.000|100|{'lower': 49, 'upper': 51}|{'lower': 49, 'upper': 49}|

최종 checkpoint 검증: ['v1/fixed-T1', 'v1/linear-T1to3']
고정 T1 v4는626688에서 조기 중단한 대조군입니다. 중간 평가를 최종 결과로 바꾸지 않습니다.
T1→3 v4도700k 이후 반대쪽 진입이40회 중0~1회로 줄고 성공이 없어서946176에서 중단했습니다. 최종1M 결과는 없습니다. v1의 완료 결과는 보존했습니다.
![동일 step 직접 정책 비교](matched_trajectories_policy.png)
