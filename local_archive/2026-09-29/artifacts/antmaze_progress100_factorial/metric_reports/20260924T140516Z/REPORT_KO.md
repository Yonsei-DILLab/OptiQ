# Progress100 경과 — 2026-09-24 23:05 KST 수집

12개 실행/4개 대기/완료0/실패0. 모든실행PID생존. Native budget v1/v2 3M, v3 4M, v4 5M 유지.

각행은최신저장40회 random-start direct-policy(random z+conditional sigma)평가다. 조건별학습step이다르며단일seed0이다.

|미로|조건|평가 step|성공|통로 사용(실패 포함)|성공 통로|
|---|---|---:|---:|---|---|
|v1|progress100_geodesic|500224|32/40|{'lower': 13, 'upper': 24, 'uncommitted': 3}|{'lower': 10, 'upper': 22}|
|v1|progress100_geodesic_no_bonus|750080|38/40|{'lower': 12, 'upper': 28}|{'lower': 11, 'upper': 27}|
|v2|progress100_euclidean|750080|36/40|{'uncommitted': 4, 'right': 36}|{'right': 36}|
|v2|progress100_euclidean_no_bonus|750080|37/40|{'uncommitted': 3, 'right': 37}|{'right': 37}|
|v3|progress100_euclidean|2000128|34/40|{'uncommitted': 5, 'left': 35}|{'left': 34}|
|v3|progress100_euclidean_no_bonus|1250048|34/40|{'right': 34, 'left': 6}|{'right': 34}|
|v3|progress100_geodesic|1500160|37/40|{'right': 37, 'uncommitted': 3}|{'right': 37}|
|v3|progress100_geodesic_no_bonus|1250048|40/40|{'right': 40}|{'right': 40}|
|v4|progress100_euclidean|2000128|33/40|{'uncommitted': 6, 'upper': 33, 'lower': 1}|{'upper': 33}|
|v4|progress100_euclidean_no_bonus|1500160|36/40|{'upper': 36, 'uncommitted': 2, 'lower': 2}|{'upper': 36}|
|v4|progress100_geodesic|1000192|16/40|{'lower': 21, 'upper': 16, 'uncommitted': 3}|{'lower': 16}|
|v4|progress100_geodesic_no_bonus|1000192|22/40|{'lower': 23, 'upper': 13, 'uncommitted': 4}|{'lower': 22}|

v3 Euclidean+B는1.25M에왼쪽30/오른쪽3으로양쪽성공했으나,1.5M/1.75M/2M평가에서는오른쪽통로방문0이다. 성공은36/40,36/40,34/40으로높게유지되므로성공률만보면이변화를놓친다.

v1 geodesic bonus OFF는500k에위23/아래15성공,750k에위27/아래11성공으로두성공경로가유지된다. v4 geodesic은양쪽통로를쓰지만성공은아래쪽뿐이며, v2와v3 geodesic은오른쪽에집중한다.

Remote saved raw NPZ를직접읽어유한좌표,랜덤초기상태,성공카운트와reward sum=100*(d0-dT)-T+B를검증하고각파일SHA256을보존했다. 큰NPZ를재전송하지않은원격읽기검증이며원자료는각raw_path에서유지된다. 새평가나학습변경없음.

![통로 추이](corridor_progress.png)
