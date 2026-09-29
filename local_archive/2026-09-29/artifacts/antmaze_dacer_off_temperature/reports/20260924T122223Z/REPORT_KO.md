# OptiQ 온도 실험 경과

Dense −nearest-distance, DACER OFF, NovelD OFF. seed0 한 개. 새로운 step penalty 보상은 적용하지 않았다.

|미로|T|학습 M / 예산 M|상태|평가 M|성공|실패 포함 통로|
|---|---:|---:|---|---:|---:|---|
|v1|3|3.008/3.008|완료|3.008|95/100|{'lower': 46, 'upper': 54}|
|v1|5|3.008/3.008|완료|3.008|0/100|{'uncommitted': 63, 'upper': 28, 'lower': 9}|
|v1|10|2.765/3.008|진행|2.750|0/40|{'uncommitted': 38, 'upper': 2}|
|v3|3|3.994/4.008|진행|3.750|40/40|{'right': 40}|
|v3|5|3.498/4.008|진행|3.250|28/40|{'right': 34, 'uncommitted': 6}|
|v3|10|4.008/4.008|진행|4.000|0/40|{'uncommitted': 27, 'left': 6, 'right': 7}|
|v4|3|5.008/5.008|완료|5.008|33/100|{'upper': 92, 'uncommitted': 8}|
|v4|5|4.153/5.008|진행|4.000|0/40|{'lower': 11, 'upper': 28, 'uncommitted': 1}|
|v4|10|3.887/5.008|진행|3.750|0/40|{'uncommitted': 7, 'upper': 22, 'lower': 11}|

중간 40회/최종100회 랜덤 시작 direct-policy 평가. 성공률과 실패 포함 통로 방문을 구분하며 동일 상태의 다중 경로를 자동으로 입증하지 않는다. 학습 step이 서로 달라 최신 결과만으로 온도별 최종 순위를 매기지 않는다.

![궤적](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_temperature/reports/20260924T122223Z/latest_trajectories.png)
![추이](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_temperature/reports/20260924T122223Z/learning_and_routes.png)
