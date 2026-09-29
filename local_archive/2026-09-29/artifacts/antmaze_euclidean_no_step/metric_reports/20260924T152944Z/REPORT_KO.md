# Euclidean progress / no bonus / no step penalty — 중간 보고

OptiQ seed0, T1, DACER OFF, NovelD OFF. 학습 source f953d28456d3800860dddb9b9cb91b6bd520ae00. 네 작업 정상 실행, failure 없음.

|미로|학습 step / budget|평가 step|직접 정책 성공|경로 분류|
|---|---:|---:|---:|---|
|v1|274,432 / 3,008,256|250112|0/40|{'uncommitted': 40}|
|v2|266,240 / 3,008,256|250112|38/40|{'right': 40}|
|v3|454,656 / 4,008,448|250112|0/40|{'left': 12, 'right': 23, 'uncommitted': 5}|
|v4|249,856 / 5,008,384|첫 평가 진행 중|미확정|미확정|

v1은 랜덤 시작, v2-v4는 원래 고정 시작/자세/속도. 직접 정책은 random latent와 conditional sigma를 포함하며 외부 탐색잡음은 없다.
v3에서 왼쪽12/오른쪽23/통과기준 미도달5. x<-8 / x>8 기준으로 분류했다. 양쪽 경로가 관찰되지만 성공0/40이며 지속 유지 여부는 아직 판단하지 않는다.
v2는 오른쪽 목표38/40 성공, 모든 rollout이 오른쪽 통로를 사용했다.
학습 중 성공횟수와 평가 성공률은 별도 지표다. 비교는 단일 seed0의 초기 학습 결과다.

그림: ../20260924T152830Z/v3_250k_trajectories.png
보고 스크립트는 frozen 학습 소스와 별도로 artifacts/antmaze_euclidean_no_step/collect_metrics.py 및 plot_v3_initial.py에 보관했다.
