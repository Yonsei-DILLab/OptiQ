# v3 teacher 후보 폭 비교

기존 teacher floor exp(-5)와0.5·1.0을 비교합니다. 알고리즘 구조와 물리 환경은 동일합니다. teacher 표본과 logq에 같은 floor를 쓰며 actor sigma 범위[-5,-1],T3,discount.999,거리 정규화 보상은 유지합니다.

| 조건 | 평가 | total step | 횟수 | 통로 진입 | 성공 통로 | 목표별 성공 경로 | 성공률 |
|---|---|---:|---:|---|---|---|---:|
|floor-exp(-5)|policy|258304|100|{'right': 84, 'left': 16}|{'right': 4}|{'right/G2': 4}|4.0%|
|floor-exp(-5)|native|258304|100|{'right': 92, 'left': 8}|{'right': 38}|{'right/G2': 38}|38.0%|
|floor-0.5|policy|258304|100|{'right': 84, 'left': 15, 'uncommitted': 1}|{'right': 8}|{'right/G2': 8}|8.0%|
|floor-0.5|native|258304|100|{'right': 89, 'left': 11}|{'right': 29}|{'right/G2': 29}|29.0%|
|floor-1|policy|258304|100|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|{'right/G2': 44}|44.0%|
|floor-1|native|258304|100|{'right': 97, 'left': 3}|{'right': 80}|{'right/G2': 80}|80.0%|

![같은 예산 직접 정책](matched_trajectories_policy.png)
![직접 정책 학습 추이](learning_curves_policy.png)

궤적 그림은 세 조건 모두 저장된 정확히 같은 step을 비교합니다. 표는 최신 결과라 step이 다를 수 있습니다. v3 원래 고정 full state, seed0 하나입니다. 직접 정책은 random z와 conditional sigma를 포함하며 외부 DACER 잡음은 없습니다. mu-only는 별도 보조 결과입니다. 각 reward profile로 보상 합·시작 상태·성공 목표·원시 파일 SHA256을 검증했습니다. 통로 진입만으로 두 성공 경로나 장기간 유지를 주장하지 않습니다.

두 새 실험의 실제250k replay 전체에서 목표 반경 진입과 goal terminal은 모두0회였습니다. 고정된 최종 정책의 반복 평가는 학습 중 수집 기록과 구분해야 합니다. 256개 환경별로는 warmup 포함1009transition을 수집한 시점입니다. 이번 방문 편중을 성공 terminal 경험 이후의 현상만으로 설명할 수는 없으며, 이 사실만으로 그 원인이나 해결책이 확정되지는 않습니다.

160k 시점의8개 실제 minibatch 로그에서 teacher ESS 평균은 floor-exp(-5): 19.27/64, floor-0.5: 30.77/64, floor-1: 33.87/64였습니다. 이 값은 전체 학습 평균이나 동일 시작 상태의 Q 진단이 아닙니다. teacher 표본의 유효 개수 증가는 확인되지만 양쪽 성공으로 이어지지는 않았습니다.
