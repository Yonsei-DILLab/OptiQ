# σ 상한 비교 — 250k 결과

유한 상한0/1/2/3의8개 완료. 무상한2개는 inf를 JSON으로 저장하는 단계에서 실패했고 본학습을 시작하지 못했다. 학습 발산으로 분류하지 않는다.

아래는 같은 초기 full state에서 매 행동 random z와 conditional sigma를 샘플링한 직접 정책100회 결과다. 초기log sigma는 모두-1이며 상한-1은 보존 대조군이다.

|환경|log σ 상한|평가|통로 진입|성공 통로|성공률|
|---|---|---|---|---|---:|
|v3|-1|policy|{'left': 11, 'right': 87, 'uncommitted': 2}|{'right': 44}|44%|
|v3|-1|native|{'right': 97, 'left': 3}|{'right': 80}|80%|
|v3|0|policy|{'left': 12, 'right': 86, 'uncommitted': 2}|{'right': 15}|15%|
|v3|0|native|{'right': 82, 'uncommitted': 18}|{'right': 33}|33%|
|v3|1|policy|{'right': 96, 'uncommitted': 2, 'left': 2}|{'right': 12}|12%|
|v3|1|native|{'right': 79, 'uncommitted': 21}|{'right': 24}|24%|
|v3|2|policy|{'right': 97, 'uncommitted': 3}|{'right': 2}|2%|
|v3|2|native|{'right': 71, 'uncommitted': 29}|{'right': 16}|16%|
|v3|3|policy|{'right': 84, 'uncommitted': 13, 'left': 3}|{'right': 4}|4%|
|v3|3|native|{'uncommitted': 72, 'right': 28}|{'right': 1}|1%|
|v4|-1|policy|{'upper': 71, 'lower': 29}|{}|0%|
|v4|-1|native|{'upper': 68, 'lower': 32}|{}|0%|
|v4|0|policy|{'upper': 95, 'lower': 5}|{}|0%|
|v4|0|native|{'upper': 99, 'uncommitted': 1}|{}|0%|
|v4|1|policy|{'lower': 39, 'upper': 61}|{}|0%|
|v4|1|native|{'uncommitted': 2, 'lower': 47, 'upper': 51}|{}|0%|
|v4|2|policy|{'lower': 64, 'upper': 34, 'uncommitted': 2}|{}|0%|
|v4|2|native|{'upper': 61, 'lower': 34, 'uncommitted': 5}|{}|0%|
|v4|3|policy|{'upper': 99, 'uncommitted': 1}|{}|0%|
|v4|3|native|{'upper': 98, 'uncommitted': 2}|{}|0%|

![v3 직접정책](v3_final_policy.png)
![v4 직접정책](v4_final_policy.png)
![학습곡선](learning_curves_policy.png)

이 실험은 공통 기본값이 아니다. v3=정규화된geodesic progress100/T3/teacherfloor1, v4=geodesic progress100/T1/teacherfloor.5, 둘 다gamma.999/DACER목표+.7·500update/NovelD OFF이다. 각 환경 내에서만 상한 효과를 비교한다.
단일seed·250k이다. 모든실패를 분모에 포함하고 진입과 성공을 구분했다. 정책·학습코드는 바꾸지 않았으며 원시rollout의SHA, 시작상태, 보상, 목표도달과 최종replay검증기록을 대조했다. 무상한의 성능 결과는 없다.
