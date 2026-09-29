# v3 teacher 후보 폭 비교

기존 teacher floor exp(-5)와0.5·1.0을 비교합니다. 알고리즘 구조와 물리 환경은 동일합니다. teacher 표본과 logq에 같은 floor를 쓰며 actor sigma 범위[-5,-1],T3,discount.999,거리 정규화 보상은 유지합니다.

| 조건 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 목표별 성공 경로 | 성공률 |
|---|---|---:|---:|---|---|---|---:|
|floor-exp(-5)|policy|258304|100|{'right': 84, 'left': 16}|{'right': 4}|{'right/G2': 4}|4.0%|
|floor-exp(-5)|native|258304|100|{'right': 92, 'left': 8}|{'right': 38}|{'right/G2': 38}|38.0%|

![같은 예산 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

궤적 그림은 세 조건 모두 저장된 정확히 같은 step을 비교합니다. 표는 최신 결과라 step이 다를 수 있습니다. v3 원래 고정 full state, seed0 하나입니다. 직접 정책은 random z와 conditional sigma를 포함하며 외부 DACER 잡음은 없습니다. mu-only는 별도 보조 결과입니다. 각 reward profile로 보상 합·시작 상태·성공 목표·원시 파일 SHA256을 검증했습니다. 통로 진입만으로 두 성공 경로나 장기간 유지를 주장하지 않습니다.
