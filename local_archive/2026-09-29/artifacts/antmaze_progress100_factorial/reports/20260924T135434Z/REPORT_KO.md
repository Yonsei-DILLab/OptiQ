# Progress100 중간 경과 — 2026-09-24 22:55 KST 수집

12개 실행,4개 대기,완료0,실패0. v3/v4 약0.77M~1.45M, v1/v2 약0.41M~0.47M. 설정/학습 변경 없이 저장 결과만 수집했다.

Random-start direct policy(random z+conditional sigma), 각40episode, 학습seed0. 학습 진행률과 아래 저장 평가 시점은 다르다.

|미로|프로필|평가 step|성공|전체 통로|성공 통로|
|---|---|---:|---:|---|---|
|v1|progress100_geodesic|250112|5/40|{'lower': 15, 'upper': 21, 'uncommitted': 4}|{'upper': 5}|
|v1|progress100_geodesic_no_bonus|250112|5/40|{'lower': 15, 'upper': 21, 'uncommitted': 4}|{'upper': 5}|
|v2|progress100_euclidean|250112|37/40|{'uncommitted': 3, 'right': 37}|{'right': 37}|
|v2|progress100_euclidean_no_bonus|250112|39/40|{'uncommitted': 1, 'right': 39}|{'right': 39}|
|v3|progress100_euclidean|1250048|33/40|{'right': 9, 'left': 30, 'uncommitted': 1}|{'left': 30, 'right': 3}|
|v3|progress100_euclidean_no_bonus|750080|10/40|{'right': 29, 'left': 11}|{'right': 10}|
|v3|progress100_geodesic|1000192|36/40|{'right': 36, 'uncommitted': 4}|{'right': 36}|
|v3|progress100_geodesic_no_bonus|750080|40/40|{'right': 40}|{'right': 40}|
|v4|progress100_euclidean|1250048|12/40|{'lower': 3, 'upper': 36, 'uncommitted': 1}|{'upper': 12}|
|v4|progress100_euclidean_no_bonus|1000192|36/40|{'upper': 39, 'lower': 1}|{'upper': 36}|
|v4|progress100_geodesic|750080|2/40|{'lower': 19, 'upper': 19, 'uncommitted': 2}|{'lower': 2}|
|v4|progress100_geodesic_no_bonus|750080|17/40|{'lower': 23, 'upper': 15, 'uncommitted': 2}|{'lower': 17}|

핵심: v3 Euclidean+B는1.25M에서 왼쪽30회/오른쪽3회 모두 성공했다. 랜덤 시작점의 서로 다른 rollout에서 관찰한 두 성공 경로이며 동일 초기 상태의 다중모드 증명은 아니다. v3 geodesic은750k40/40에 도달했으나 오른쪽만 사용한다.

v4 Euclidean은 성공률 상승과 함께 아래 통로 사용이 감소했다. B ON은500k 아래9→1.25M 아래3; B OFF는500k 아래10→1M 아래1이다. v4 geodesic은750k에 위/아래19/19(B ON),15/23(B OFF)로 양쪽을 사용하지만 성공은 아직 아래쪽뿐이다.

v1 geodesic은250k에서두조건모두5/40성공, 위21/아래15를 사용하고 성공은위쪽만이다. v2 Euclidean은250k에서B ON37/40,OFF39/40이지만오른쪽목표뿐이다.

결론: 보상 확대 후 목표 도달 학습은 진행된다. 성공률 상승이 두 경로 유지를 보장하지 않으며, 현재 두 성공 경로가 확인된 조건은v3 Euclidean+B다. 단일seed/중간평가라 장기 유지 및 B의 인과효과를 단정하지 않는다.

모든 raw NPZ 전송SHA256,유한성,40개랜덤시작상태,성공카운트,100*(d0-dT)-T+B 보상합을 검증했다. 비교는같은예산을기준으로해야하며다른step의프로필을직접서열화하지않는다.

![v1](v1_trajectories.png)
![v2](v2_trajectories.png)
![v3](v3_trajectories.png)
![v4](v4_trajectories.png)
