# 고정 latent 64개와 기존 random latent 비교

각 환경의 대조군과 보상·temperature·teacher floor·모델·초기 가중치·학습 예산은 같습니다. 기존 코드에 있는 finite64 설정만 사용합니다. 매 행동마다 동일한 64개 codebook에서 균등하게 다시 선택하며, episode 동안 z를 고정하지 않습니다.

|환경|prior|mode|total step|평가 수|통로 진입|성공 통로|
|---|---|---|---:|---:|---|---|
|v3|random-prior|policy|258304|100|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|
|v3|random-prior|native|258304|100|{'right': 97, 'left': 3}|{'right': 80}|
|v3|fixed64-prior|policy|258304|100|{'right': 92, 'left': 4, 'uncommitted': 4}|{'right': 15}|
|v3|fixed64-prior|native|258304|100|{'right': 90, 'uncommitted': 4, 'left': 6}|{'right': 56}|
|v4|random-prior|policy|258304|100|{'upper': 71, 'lower': 29}|{}|
|v4|random-prior|native|258304|100|{'upper': 68, 'lower': 32}|{}|
|v4|fixed64-prior|policy|258304|100|{'uncommitted': 2, 'lower': 71, 'upper': 27}|{}|
|v4|fixed64-prior|native|258304|100|{'lower': 75, 'upper': 25}|{'upper': 1}|

![동일 step 직접 정책 비교](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

그림은 각 환경에서 두 조건의 동일한 step을 비교합니다. 표는 조건별 최신 평가로 step이 다를 수 있습니다. v3·v4 모두 원래 동일한 위치·자세·속도에서 시작합니다. 직접 정책은 conditional sigma를 포함하며 외부 DACER 잡음을 넣지 않습니다. mu-only는 별도 보조 결과입니다. 첫 통로 진입 수와 해당 episode의 목표 성공 수를 분리합니다. 실패를 분모에서 제외하지 않으며 한 seed의 250k 결과만으로 장기 유지나 재현성을 주장하지 않습니다.

실제 8update 사전검사, sampling/serialization, 초기 parameter hash, 고정 codebook, frozen 알고리즘 파일 9개를 검사했습니다. 보고서는 raw 궤적의 full start·보상합·성공 목표·SHA256을 검증하며 학습 결과를 덮어쓰지 않습니다. finite의 결정론적 보조 평가는 component0_mu로 저장되며 z=0 평가로 부르지 않습니다.

완료 및 최종 replay 검증된 조건: [{'task': 'v3', 'condition': 'random-prior', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-250k-s0'}, {'task': 'v3', 'condition': 'fixed64-prior', 'id': 'v3-optiq-startnorm-geodesic-T3-teacherfloor1-fixed64-250k-s0'}, {'task': 'v4', 'condition': 'random-prior', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-250k-s0'}, {'task': 'v4', 'condition': 'fixed64-prior', 'id': 'v4-optiq-geodesic-T1-teacherfloor0.5-fixed64-250k-s0'}]
보고 코드는 학습 시작 후 만들어진 후처리이며 SHA와 학습 source를 results.json에 구분했습니다.
