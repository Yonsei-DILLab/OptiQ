# 동일 1M 시점 보상 비교

T1, DACER OFF, NovelD OFF. 1,000,192 transitions, 랜덤 시작 direct-policy 40회, seed0. 시작 full-state와 native 알고리즘 설정 일치 확인.

|미로|보상|성공|통로 사용|평균 최대 출발점 이격(m)|평균 목표 최근접 거리(m)|
|---|---|---|---|---:|---:|
|v3|dense|0/40|{'uncommitted': 15, 'left': 21, 'right': 4}|10.62|6.63|
|v3|progress_euclidean|0/40|{'uncommitted': 40}|2.40|14.57|
|v4|dense|0/40|{'lower': 5, 'upper': 32, 'uncommitted': 3}|10.41|7.47|
|v4|progress_euclidean|0/40|{'uncommitted': 40}|2.46|15.14|

v3 통로 기준 x<-8 / x>8, v4 x=-4 통과 시 y>2 / y<-2. 통로 통과와 성공을 구분. v3 새 Geodesic은 수집 당시 1M 평가 미완료라 동일 1M 표에 포함하지 않음. 새 성공보상 ON/OFF는 1M 전 목표 도달이 없어 동일 궤적을 생성했다. v4 Geodesic은 1M 성공0/40, 통로0/40, 평균최대이격2.54m.

원시 return은 보상 정의가 달라 직접 비교하지 않음. 한 시드 결과이며 최종 성능은 미확정.

![비교](comparison_1m.png)
