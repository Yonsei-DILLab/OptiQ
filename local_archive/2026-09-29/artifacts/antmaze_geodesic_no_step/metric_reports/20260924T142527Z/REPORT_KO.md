# Geodesic B=0 / step penalty=0 경과

2026-09-24 23:25 KST. 새 4개 실행 중, 실패0. 학습 source4d757159ef84e849250a6d5111f9931b2b392b21.

보상100*(d현재-d다음), T1, DACER/NovelD OFF. 아래 평가는 모두250112 interactions, seed0 한 정책의40개 random-start direct-policy rollout이다. 조건부 sigma를 포함하고 외부 잡음은 없다.

|미로|현재 interactions|평가 성공|통로 방문(실패 포함)|성공 통로|
|---|---:|---:|---|---|
|v1|327680|3/40|{'lower': 16, 'upper': 22, 'uncommitted': 2}|{'upper': 3}|
|v2|278528|37/40|{'uncommitted': 2, 'right': 38}|{'right': 37}|
|v3|405504|19/40|{'right': 40}|{'right': 19}|
|v4|266240|0/40|{'lower': 19, 'upper': 21}|{}|

v1/v4는 양쪽 통로를 방문하지만 v1 성공은위쪽3회뿐이고 v4 성공은0회다. v2/v3는오른쪽으로집중한다. 통로분류는v1/v4의x=-4첫교차y>2/ y<-2, v2 x±4, v3 x±8 방문기준이다.

같은250k 기존 geodesic B0 step-1 결과: v1 성공5/40(위21/아래15/미진입4),v3 성공16/40(오른쪽36/미진입4),v4 성공0/40(위20/아래18/미진입2). 초기한seed평가에서step penalty제거가경로다양성을개선했다는증거는아직없다.

원격 저장NPZ의유한좌표,랜덤초기상태,성공카운트및sum(r)=100*(d0-dT)를검증했다. raw SHA256과경로는analysis.json에보관. 원자료는서버에유지, 새평가나학습설정변경없음.
