# v3 teacher floor1: 목표 도달·경로 유지 확인

보상·T3·할인율.999·teacher floor1·모델을 그대로 유지하며 학습 예산만 늘립니다. 같은 seed0의 짧은/긴 실험은 독립 seed가 아닙니다. 원래 고정 full state에서 평가하며 실패를 분모에 포함합니다. 성공한 양쪽 경로와 이후 유지를 확인하기 전에는 목표 달성으로 보고하지 않습니다.

|조건|mode|total step|평가 수|통로 진입|성공 통로|
|---|---|---:|---:|---|---|
|floor1-short|policy|258304|100|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|
|floor1-short|native|258304|100|{'right': 97, 'left': 3}|{'right': 80}|
|floor1-long|policy|650240|40|{'right': 40}|{'right': 29}|
|floor1-long|native|650240|40|{'right': 40}|{'right': 25}|

최종 전체 checkpoint 검증: ['floor1-short']
같은 설정의 저장 궤적 prefix 일치 검사: 10
최신 그림은 조건별 step을 표시합니다. 초기/중간40회를 최종100회로 부르지 않습니다.

장기 확인 실험은 반대쪽 경로 상실이 반복되어 조기 중단했습니다. 계획한 1M 완료나 최종100회 평가 결과로 해석하지 않습니다. 로그·중간 정책·중단 근거를 보존했습니다.
중단 직전 기록 step: {'v3-optiq-startnorm-geodesic-T3-teacherfloor1-1m-s0': 696320}
