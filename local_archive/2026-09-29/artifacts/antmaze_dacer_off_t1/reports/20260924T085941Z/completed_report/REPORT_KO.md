# DACER OFF: 완료 결과와 진행 중 결과

OptiQ, T=1, dense reward, NovelD OFF, 각 미로 seed0. 주 결과는 random-start direct-policy(random z + conditional sigma) 평가다. 성공/실패 모두 포함해 통로를 분류했다.

|미로|상태|학습 step / 예산|평가 step|성공|실패 포함 통로|
|---|---|---:|---:|---:|---|
|v1|completed|3,008,256 / 3,008,256|3,008,256|99/100|{'lower': 34, 'upper': 65, 'uncommitted': 1}|
|v2|completed|3,008,256 / 3,008,256|3,008,256|94/100|{'right': 94, 'uncommitted': 6}|
|v3|completed|4,008,448 / 4,008,448|4,008,448|96/100|{'right': 96, 'uncommitted': 4}|
|v4|training|4,083,712 / 5,008,384|4,000,000|33/40|{'upper': 34, 'uncommitted': 6}|

## 완료된 정책: 평가 방식별 성공

|미로|direct random|mu-only random|zero_z random|direct 동일 전체 상태|
|---|---:|---:|---:|---:|
|v1|99/100|97/100|91/100|99/100|
|v2|94/100|94/100|95/100|73/100|
|v3|96/100|90/100|69/100|100/100|

동일 상태 평가는 한 개의 선택된 전체 초기 상태를 반복한 것이다. 랜덤 시작점 전체의 성능과 같지 않다. 랜덤 시작점에서 두 통로를 쓰는 것과 같은 상태에서 두 통로로 나뉘는 것은 구분한다.

## 완료된 DACER ON 대조군과 비교

|미로|ON final 성공/100|OFF final 성공/100|ON return|OFF return|
|---|---:|---:|---:|---:|
|v1|100|99|-607.2|-628.9|
|v2|94|94|-598.0|-574.4|
|v3|96|96|-2160.9|-1701.5|

시드 하나 및 평가 100회 결과이므로 작은 차이를 일반적인 개선/악화로 단정하지 않는다. 설정 차이는 analysis.json에 전부 보존했다.

## 이전에 완료된 dense + NovelD OFF 실험 요약

아래는 각 미로 최종 direct-policy, random-start 100회 성공률(%). 방법마다 native sampling을 사용하며 원시 검증 결과를 재사용했다. 서로 다른 정책의 궤적을 합쳐 다양성을 계산하지 않았다.

|방법/설정|v1|v2|v3|v4|
|---|---:|---:|---:|---:|
|T=0.01|98|92|0|86|
|T=1|100|94|96|84|
|10->1|100|98|100|83|
|10->0.25|99|99|100|78|
|10->0.5|100|89|100|65|
|SAC|0|100|3|0|
|DIPO|98|88|0|84|
|MFPO|23|83|83|62|

OptiQ의 위 이전 실험은 DACER ON이다. 현재 OFF 실험의 미완료 결과를 최종 비교표에 섞지 않았다.

## 보존 및 검증

config·평가 summary/원시 NPZ·checkpoint 검증 증명·result JSON을 로컬에 보존했다. 전송 SHA256, 원시 좌표/목표 도달/시작 상태/padding/유한성/dense 거리 보상 합을 검증했다. 완료 학습의 전체 replay/model checkpoint는 서버에 보존되며 서버 readback 및 dense replay 검증을 통과했다. 이 보고는 새 학습이나 rollout을 수행하지 않는다.

![전체 궤적](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_t1/reports/20260924T085941Z/completed_report/all_trajectories.png)

![학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_t1/reports/20260924T085941Z/completed_report/learning_curves.png)

![통로 선택 추이](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_t1/reports/20260924T085941Z/completed_report/corridor_history.png)
