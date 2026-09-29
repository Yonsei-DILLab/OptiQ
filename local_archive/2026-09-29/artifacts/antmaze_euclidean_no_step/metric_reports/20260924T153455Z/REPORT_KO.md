# Euclidean no-bonus / no-step-penalty 진행 보고

사용자 판단 반영: v2의 한쪽 경로 집중은 이번 비교의 문제 항목으로 다루지 않는다. v1/v3/v4의 경로 다양성과 성공을 중심으로 확인한다.

학습 source f953d28456d3800860dddb9b9cb91b6bd520ae00. 네 작업 정상 실행 중. OptiQ seed0, T1, DACER OFF, NovelD OFF.

|환경|학습 진행|최근 평가|성공|경로|
|---|---:|---:|---:|---|
|v1|516,096/3,008,256|500,224|0/40|{'uncommitted': 38, 'upper': 2}|
|v2|512,000/3,008,256|500,224|40/40|{'right': 40}|
|v3|761,856/4,008,448|750,080|37/40|{'right': 39, 'uncommitted': 1}|
|v4|450,560/5,008,384|250,112|0/40|{'upper': 28, 'uncommitted': 3, 'lower': 9}|

v3는250112step 왼쪽12/오른쪽23/분기미통과5, 성공0/40이었다. 500224step에서 왼쪽0/오른쪽39/미통과1, 성공16/40으로 바뀌었고, 750080step도 왼쪽0/오른쪽39/미통과1, 성공37/40이었다. 이번40회 샘플에서는 왼쪽 경로가 다시 관찰되지 않는다. 따라서 성공률 향상과 두 경로 유지가 분리되어 나타났다. 원인을 이 관찰만으로 단정하지 않는다.
v1은500224step에도 성공0/40,38회는분기통과미달,2회만위통로에진입했다. v4는250112step 위28/아래9/미통과3이며성공0/40;아직500k평가가없어이후유지여부미확정.

v1 random starts; v2-v4 original fixed full state. Direct-policy sampling includes random z and conditional sigma; no external exploration noise. Path gate thresholds: v1/v4 first x=-4 crossing at y>2 or y<-2; v2 x<-4 or x>4; v3 x<-8 or x>8. Success means raw goal ID>0. Each reported row uses40rollouts from one seed0 checkpoint. All returns independently checked against100*(d_start-d_end), rawSHA256 inanalysis.json.
