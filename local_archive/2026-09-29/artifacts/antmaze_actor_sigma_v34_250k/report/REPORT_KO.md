# Conditional sigma 상한·초기값 비교

log sigma 상한과 초기값을 함께 −1/−2/−3으로 비교합니다. 하한−5, random latent, teacher 탐색폭, 보상·모델·학습률·총 transition·업데이트 수는 같습니다. 상한 효과와 초기화 효과를 분리한 실험은 아닙니다.

|환경|조건|mode|total step|평가 수|통로 진입|성공 통로|
|---|---|---|---:|---:|---|---|
|v3|capm1|policy|258304|100|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|
|v3|capm1|native|258304|100|{'right': 97, 'left': 3}|{'right': 80}|
|v3|capm2|policy|258304|100|{'right': 91, 'left': 9}|{'right': 15}|
|v3|capm2|native|258304|100|{'right': 96, 'uncommitted': 2, 'left': 2}|{'right': 25}|
|v3|capm3|policy|258304|100|{'right': 86, 'left': 10, 'uncommitted': 4}|{'right': 14}|
|v3|capm3|native|258304|100|{'right': 88, 'left': 5, 'uncommitted': 7}|{'right': 15}|
|v4|capm1|policy|258304|100|{'upper': 71, 'lower': 29}|{}|
|v4|capm1|native|258304|100|{'upper': 68, 'lower': 32}|{}|
|v4|capm2|policy|258304|100|{'lower': 96, 'upper': 3, 'uncommitted': 1}|{}|
|v4|capm2|native|258304|100|{'lower': 97, 'upper': 3}|{}|
|v4|capm3|policy|258304|100|{'lower': 91, 'upper': 9}|{}|
|v4|capm3|native|258304|100|{'lower': 92, 'upper': 6, 'uncommitted': 2}|{}|

![동일 step 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

주 그림은 모든 조건의 동일 step만 비교합니다. 표는 조건별 최신 평가입니다. 직접 정책은 학습된 conditional sigma를 실제로 샘플링하며 외부 DACER 잡음을 넣지 않습니다. mu-only는 보조 결과입니다. v3/v4는 원래 동일 full state로 시작합니다. 실패를 분모에서 제외하지 않으며, 통로 진입만으로 성공이나 장기 유지를 주장하지 않습니다. 단일 seed이며 원시 궤적의 보상·full start·목표·SHA, 최종 replay와 초기 parameter 동등성을 검증합니다.
최종 검증된 조건: [{'task': 'v3', 'condition': 'capm1', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-250k-s0'}, {'task': 'v3', 'condition': 'capm2', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-capm2-250k-s0'}, {'task': 'v3', 'condition': 'capm3', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-capm3-250k-s0'}, {'task': 'v4', 'condition': 'capm1', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-250k-s0'}, {'task': 'v4', 'condition': 'capm2', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-capm2-250k-s0'}, {'task': 'v4', 'condition': 'capm3', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-capm3-250k-s0'}]
보고 코드는 학습 후 작성한 후처리이며 학습 source와 SHA를 별도로 저장했습니다.
