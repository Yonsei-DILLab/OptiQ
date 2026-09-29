# v3 보정 보상: 할인율·장기 유지 비교

보상·T3·모델은 같고 .99999 조건에서 할인율만 바뀝니다. 같은 seed0의 짧은/긴 실험은 독립 seed가 아닙니다. 원래 고정 full state에서 평가하며 실패를 분모에 포함합니다. 성공한 양쪽 경로와 이후 유지를 확인하기 전에는 목표 달성으로 보고하지 않습니다.

|조건|mode|total step|평가 수|통로 진입|성공 통로|
|---|---|---:|---:|---|---|
|gamma999-short|policy|258304|100|{'right': 84, 'left': 16}|{'right': 4}|
|gamma999-short|native|258304|100|{'right': 92, 'left': 8}|{'right': 38}|
|gamma999-long|policy|550144|40|{'right': 40}|{'right': 35}|
|gamma999-long|native|600064|40|{'right': 40}|{'right': 35}|
|gamma99999-long|policy|550144|40|{'right': 40}|{'right': 35}|
|gamma99999-long|native|600064|40|{'right': 40}|{'right': 35}|

최종 전체 checkpoint 검증: ['gamma999-short']
같은 설정의 저장 궤적 prefix 일치 검사: 10
최신 그림은 조건별 step을 표시합니다. 초기/중간40회를 최종100회로 부르지 않습니다.

두 장기 실험은 반대쪽 경로 상실이 반복되어 조기 중단했습니다. 계획한 1M 완료나 최종100회 평가 결과로 해석하지 않습니다. 로그·중간 정책·중단 근거를 보존했습니다.
중단 직전 기록 step: {'v3-optiq-startnorm-geodesic-gamma999-T3-H0.7-i500-1m-s0': 598016, 'v3-optiq-startnorm-geodesic-gamma99999-T3-H0.7-i500-1m-s0': 598016}
