# 한쪽 경로로 집중되는 시점과 원인 점검

대상: 현재 step-penalty ON, geodesic, 보상 100*(d_current-d_next)-1, B=0, T=1, DACER/NovelD OFF. v3/v4 seed0. 학습을 변경하지 않고 저장 정책을 동일한 원래 전체 초기 상태에서 재평가했다. 각 checkpoint direct-policy40회, native40회; 이 보고의 주 결과는 direct-policy(random z+conditional sigma)다.

## 같은 고정 상태에서의 변화

|환경|체크포인트|통로 선택|성공|
|---|---:|---|---:|
|v3|250,112|{'right': 40}|22/40|
|v3|750,080|{'right': 40}|39/40|
|v3|3,000,064|{'right': 40}|40/40|
|v4|250,112|{'upper': 17, 'lower': 22, 'uncommitted': 1}|0/40|
|v4|500,224|{'lower': 40}|1/40|
|v4|1,000,192|{'lower': 40}|38/40|
|v4|1,500,160|{'lower': 40}|40/40|
|v4|2,500,096|{'lower': 40}|40/40|

v4에서는250k에 실제 양방향 탐색이 있었으나500k부터 위쪽 선택이40회에서관측되지않았다. 두 성공 경로를 유지했다는 뜻은 아니다:250k에는양쪽모두실패했다. v3는첫저장250k부터오른쪽40/40이다.250k이전과저장시점사이의모든정책을관측한것은아니다.

## v4: 초기의 실제 수행 능력이 달랐다

|250k에서 선택한 통로|평가 수|마지막 geodesic 거리|누적 보상|할인 누적 보상 gamma=.99|
|---|---:|---:|---:|---:|
|upper|17|10.54m|11.3|161.5|
|lower|22|6.44m|421.8|337.1|

두 그룹 모두원래같은전체상태에서출발했고700step에서timeout됐다. Step penalty 합은동일하게-700이다. 따라서누적보상차이는progress차이에서발생했다. 아래쪽으로간rollout이더멀리진행했고할인보상도높았다. 이는더잘배운쪽을강화하는설명과일치한다. 단,통로로조건화한실현return이지critic예측Q나강제방향개입실험은아니다. Timeout이후가치는포함하지않으므로critic Q와직접동일시하지않는다.

## v3: 시작점의 거리 보상부터 대칭이 아니다

원래원점에서왼쪽위목표의geodesic거리는17.98646m,오른쪽아래는16.97056m다. d=min(distance_to_each_goal)이므로오른쪽아래목표가현재기준이다. 가상의작은위치변화(실제로학습된action을재현한것이아님)를대조하면 (0,0)→(-.1,+.1)은nearest distance가17.11198m로늘어나고, (+.1,-.1)은16.82914m로줄어든다. 전자는progress항-14.142,후자는+14.142다. Step penalty는양쪽동일-1이다. v4는원점에서두목표거리17.65686m로동일하므로이기하적편향을그대로적용할수없다. geometry_probe.json에원자료와upstream geometry SHA를저장했다.

## 정책 업데이트와 확인 범위

Frozen source의actor는 softmax(Q/T - beta*log proposal density) 가중치를Direct GMM NLL로모사한다. 이번critic은plain TD이며학습로그의backup entropy term은0이다. 통로별점유율이나두목표방문비율을유지하는목표는없다. T는1로일정하며sigma파라미터평균은약.34~.37로유지됐다. 행동무작위성이남아있어도같은장기경로로이동할수있으며,sigma의갑작스러운0수렴이이번전환의설명은아니다.

현재v4에서아래쪽의상대적진행우위와이후선택집중은관측됐다. 최초방향편향의원인을초기화/동역학/critic오차별로분리하지않았다. 현재실험의full replay/전체training XY는완료전이므로반대편경험의버퍼퇴출을이번run에서확인했다고주장하지않는다.

예전negative-distance dense 실험은별개다. 당시v3에서는학습방문이오른쪽으로쏠린뒤왼쪽진입동일상태의후속rollout능력퇴행,최종replay왼쪽gate표본0개,같은상태/행동의critic오류를실제로검증했다. 이과거증거를이번progress-reward run의측정값으로섞지않는다. 이전보고서: ../antmaze_dacer_off_t1/v3_route_collapse_audit/controlled_continuation/REPORT_KO.md

## 보관·검증

학습source0751e86a547dc8f1d82e4861c2931cc6f9cf04b8; 재평가source061e9888e1f3d0e59b5037441fa146c1eff25e2a. 추가역사평가6개모두완료및checkpoint전후SHA/파라미터불변/동일시작상태/보상재계산검증통과. 로컬수집파일SHA256검증통과. 일부평가의W&B인증실패는별도기록돼있고검증된로컬원자료는보존됐다. 기존학습프로세스및하이퍼파라미터변경없음.

![v4 fixed history](v4_fixed_history.png)
