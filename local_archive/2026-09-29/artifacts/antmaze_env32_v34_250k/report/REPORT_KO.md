# 병렬 환경 수 비교: 총 업데이트 비율 유지

256개 수집 후 8회 업데이트와 32개 수집 후 1회 업데이트를 비교합니다. 총 transition·업데이트 수·batch4096·모델·초기 가중치·보상·평가 시점은 같고, 병렬 수집과 업데이트 블록 크기가 달라집니다. 각 환경은 최종 예산에서 1,009회 또는 8,072회 step을 수행하므로 replay 구성과 각 궤적 중 정책 변화 속도는 다릅니다.

|환경|조건|mode|total step|평가 수|통로 진입|성공 통로|
|---|---|---|---:|---:|---|---|
|v3|env256-update8|policy|258304|100|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|
|v3|env256-update8|native|258304|100|{'right': 97, 'left': 3}|{'right': 80}|
|v3|env32-update1|policy|258304|100|{'right': 88, 'uncommitted': 5, 'left': 7}|{'right': 48}|
|v3|env32-update1|native|258304|100|{'right': 94, 'left': 5, 'uncommitted': 1}|{'right': 48}|
|v4|env256-update8|policy|258304|100|{'upper': 71, 'lower': 29}|{}|
|v4|env256-update8|native|258304|100|{'upper': 68, 'lower': 32}|{}|
|v4|env32-update1|policy|258304|100|{'lower': 100}|{}|
|v4|env32-update1|native|258304|100|{'lower': 100}|{}|

![동일 step 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

v3·v4는 원래 동일한 full state에서 시작하며, 직접 정책은 random latent와 conditional sigma를 포함합니다. 외부 DACER 행동잡음은 평가하지 않습니다. mu-only는 보조 결과입니다. 한 seed이며, 통로 진입만으로 성공 경로 다양성이나 장기 유지를 주장하지 않습니다. 원자료의 보상 합·시작 상태·목표 도달·SHA와 최종 replay를 검증합니다.
검증된 최종 조건: [{'task': 'v3', 'condition': 'env256-update8', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-250k-s0'}, {'task': 'v3', 'condition': 'env32-update1', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-env32-250k-s0'}, {'task': 'v4', 'condition': 'env256-update8', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-250k-s0'}, {'task': 'v4', 'condition': 'env32-update1', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-env32-250k-s0'}]
보고 코드는 학습 후 작성한 후처리이며 SHA와 학습 source를 results.json에 구분합니다.
