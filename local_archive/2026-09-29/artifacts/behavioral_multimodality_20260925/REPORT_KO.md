# 저장된 AntMaze 결과의 동일 초기 상태 행동 다양성

최종 체크포인트마다 정책을 직접 샘플링한 저장 rollout 100회를 재집계했다. 네 방법의 각 미로 내 100회는 동일한 qpos/qvel에서 시작한다. 학습 seed는 각 1개다.

## 동일 상태 진단

|미로|방법|성공/100|성공 행동 종류|성공 경로|방문 0.5m 격자|평균 return|
|---|---|---:|---:|---|---:|---:|
|v1|Gaussian SAC|0|0|{"failure": 100}|43|-3502.7|
|v1|MFPO flow|32|1|{"G1/lower": 32, "failure": 68}|111|-1718.5|
|v1|DIPO diffusion|94|1|{"G1/lower": 94, "failure": 6}|114|-803.2|
|v1|OptiQ/iBOLT|100|1|{"G1/lower": 100}|88|-1014.8|
|v2|Gaussian SAC|97|1|{"failure": 3, "goal(8,0)/central-corridor": 97}|57|-210.2|
|v2|MFPO flow|100|1|{"goal(8,0)/central-corridor": 100}|38|-142.7|
|v2|DIPO diffusion|58|1|{"failure": 42, "goal(8,0)/central-corridor": 58}|105|-1346.5|
|v2|OptiQ/iBOLT|83|1|{"failure": 17, "goal(8,0)/central-corridor": 83}|70|-683.4|
|v3|Gaussian SAC|2|1|{"G2/passage-y-8": 2, "failure": 98}|130|-3006.9|
|v3|MFPO flow|100|1|{"G2/passage-y-8": 100}|104|-688.9|
|v3|DIPO diffusion|0|0|{"failure": 100}|99|-3518.6|
|v3|OptiQ/iBOLT|16|1|{"G1/passage-y+4": 16, "failure": 84}|263|-9344.4|
|v4|Gaussian SAC|0|0|{"failure": 100}|233|-9660.0|
|v4|MFPO flow|44|1|{"G2/lower-entry/lower-outer": 44, "failure": 56}|133|-9283.8|
|v4|DIPO diffusion|96|1|{"G2/lower-entry/lower-outer": 96, "failure": 4}|209|-2410.8|
|v4|OptiQ/iBOLT|70|1|{"G1/upper-entry/upper-outer": 70, "failure": 30}|187|-7033.5|

성공 행동 종류는 목표 ID와 미로 통로 교차를 결합한 사전 정의 경로 범주 수다. 성공이 없는 경우 0으로 기록한다. 방문 격자는 학습 exploration이 아닌 저장 평가 궤적에서 센 것이다.

## 랜덤 시작 저장 평가

|미로|방법|성공/100|평균 return|
|---|---|---:|---:|
|v1|Gaussian SAC|0|-3290.0|
|v1|MFPO flow|23|-1246.5|
|v1|DIPO diffusion|98|-444.4|
|v1|OptiQ/iBOLT|100|-607.2|
|v2|Gaussian SAC|100|-238.3|
|v2|MFPO flow|83|-1153.3|
|v2|DIPO diffusion|88|-713.4|
|v2|OptiQ/iBOLT|94|-598.0|
|v3|Gaussian SAC|3|-3313.6|
|v3|MFPO flow|83|-3063.6|
|v3|DIPO diffusion|0|-4063.6|
|v3|OptiQ/iBOLT|96|-2160.9|
|v4|Gaussian SAC|0|-6692.3|
|v4|MFPO flow|62|-5383.1|
|v4|DIPO diffusion|84|-2929.8|
|v4|OptiQ/iBOLT|84|-3346.3|

이 랜덤 시작 평가는 당시의 역사적 평가 override다. v2-v4의 원래 학습 reset은 고정 원점이므로 현재 matched-start primary 성능으로 인용하지 않는다.

## 해석

이 네 방법의 동일 상태 최종 정책에서 관찰된 성공 경로는 방법·미로별 최대 1개다. 따라서 이 비교군만으로 iBOLT의 성공 행동 다봉성을 주장할 수 없다. 별도 과거 OptiQ v1 dense+NovelD 0.01 저장 정책의 동일 상태 100회에서는 위/아래 성공 경로가 71/29로 확인되었다. 같은 모델의 mu-only 동일 상태 평가도 75/21 성공, 4회 실패여서 conditional sigma 잡음만의 효과로 설명되지 않는다. 다만 자체 환경 포트·학습 설정이 달라 이 표의 비교군과 합산하거나 인과 비교하지 않는다.

GMM40의 target fitting 결과와 이 AntMaze 정책은 동일한 target 또는 체크포인트 쌍이 아니다. 따라서 target fitting → 행동 → RL 성능의 세 단계를 하나의 인과 실험으로 연결했다고 주장할 수 없다.

Gaussian SAC는 Gaussian 정책으로 Boltzmann 대상에 대한 reverse-KL형 목적을 사용한다. 이를 별도의 동일 actor reverse-KL ablation으로 세지 않는다. MFPO는 flow matching 방법이며 reverse-KL 대조군으로 표기하지 않는다. 따라서 요청한 네 범주 중 독립된 reverse-KL actor 대조군의 AntMaze 체크포인트/rollout은 이 저장 비교군에 없다.

그림은 DDiffPG 선행연구의 궤적 오버레이·목표/경로 구분·방문 밀도(100회 상한)·이진 방문격자 방식에 맞췄다. DTW 계층 clustering은 성공 경로가 여러 개 나왔을 때 보조 검증으로 적용할 수 있으며, 현재 표에서 1개 경로를 인위적으로 더 나누지 않았다.

선행연구: https://supersglzc.github.io/projects/ddiffpg/ (trajectory modes, density maps, binary state coverage, DTW 계층 군집).

## 검증

입력은 수집 당시 SHA256 및 현재 SHA256와 일치한다. 원시 좌표 유한성, padding, 목표 반경, dense return의 거리합, 체크포인트 readback sidecar, 동일 전체 초기 상태를 재확인했다. 모델·원본 자료·진행 중 작업은 변경하지 않았다. 정확한 run/source SHA와 입력 해시는 results.json에 기록했다.
