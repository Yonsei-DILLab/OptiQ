# Dense OptiQ 체크포인트: 새로운 랜덤 시작 재평가

기존 저장 궤적 재집계가 아니라, 저장된 학습 완료 체크포인트를 복원해 새로 7,000 episode를 rollout했다. 학습은 수행하지 않았다.
이전 포트 v1은 위쪽 468, 아래쪽 517, 실패 15로 두 성공 경로를 고르게 사용했다.
공식 환경 NovelD OFF v1도 위쪽 21회가 관찰되어 두 경로가 존재한다. 다만 아래쪽 901회로 성공의 97.7%가 한쪽에 편중됐다.
매 episode마다 xy를 [-2,2]×[-2,2]에서 독립 균등 추출하고 원래 초기 자세·속도를 유지했다. policy와 native에 같은 시작 상태 묶음을 사용했다. 단일 학습 seed0. 고정 시작 결과는 이 보고서에 섞지 않았다.
policy=fresh random z+conditional sigma. native=fresh random z mu-only. 외부 DACER 행동잡음 및 NovelD 평가보상 없음.

|모델|평가 모드|성공/평가수|성공 경로 및 실패|
|---|---|---:|---|
|legacy-v1|policy|985/1000|G1/lower: 517, G1/upper: 468, failure: 15|
|legacy-v1|native|946/1000|G1/lower: 485, G1/upper: 461, failure: 54|
|official-v1|policy|922/1000|G1/lower: 901, failure: 78, G1/upper: 21|
|official-v1|native|912/1000|failure: 88, G1/lower: 889, G1/upper: 23|
|legacy-v2|policy|412/500|goal(8,0)/central-corridor: 412, failure: 88|
|legacy-v2|native|414/500|goal(8,0)/central-corridor: 414, failure: 86|
|legacy-v3|policy|496/500|G1/passage-y+4: 496, failure: 4|
|legacy-v3|native|490/500|G1/passage-y+4: 490, failure: 10|
|legacy-v4|policy|364/500|G1/upper-entry/upper-outer: 364, failure: 136|
|legacy-v4|native|359/500|G1/upper-entry/upper-outer: 359, failure: 141|

## 시작 위치와 선택 경로

같은 랜덤 시작 분포에서도 시작 좌표에 따라 경로 선택이 달라질 수 있다. 아래는 v1 direct-policy에서 초기 y의 부호별 결과다. 위치별 조건부 선택과 전체 분포의 경로 다양성을 구분한다.

|모델|시작 영역|평가수|+y 성공|−y 성공|실패|
|---|---|---:|---:|---:|---:|
|legacy-v1|start_y_positive|492|458|25|9|
|legacy-v1|start_y_negative|508|10|492|6|
|official-v1|start_y_positive|492|21|401|70|
|official-v1|start_y_negative|508|0|500|8|

## 검증과 해석 범위

- 체크포인트 SHA256, 복원된 actor/critic/optimizer 값 일치 및 평가 후 불변을 확인했다.
- 저장 NPZ SHA256, 모든 action/xy 유한성 및 action 범위, 실제 goal 반경 도달, dense 거리 보상 합계를 로컬에서 다시 검증했다.
- 평가 시작점은 모두 서로 다르며 두 평가 모드에서 전체 qpos/qvel이 정확히 같다.
- 이전 v2-v4의 학습 시작은 고정이었다. 여기서는 사용자 요청에 따라 평가만 랜덤 시작으로 변경했다. 따라서 초기상태 분포 이동이 포함된다.
- 공식 환경 dense OFF v2-v4는 당시 final-only 저장 전 중단되어 모델이 남아 있지 않았다. 재평가한 v2-v4는 이전 포트 + NovelD0.01 완료 모델이다.
- 각기 다른 frozen 환경 구현과 학습조건을 그대로 사용했다. NovelD 단독 인과효과나 여러 학습 seed의 재현성을 주장하지 않는다.
- Wilson 95% 구간은 저장된 단일 정책의 rollout 변동에 관한 구간이며 학습 seed 간 불확실성을 나타내지 않는다.
- 랜덤 초기상태 분포에서의 여러 경로는 유효한 행동 다양성이다. 이것만으로 같은 state의 action density가 다봉이라는 주장을 하지는 않는다.
- CPU-only 네트워크 추론과 MuJoCo rollout이다. 학습 GPU/체크포인트/현재16개 캠페인 설정은 변경하지 않았다.
- initial_xy seed sequence 및 정책 난수는 provenance.json에 기록했다. 전체 raw actions/xy/initial states를 보관한다.

## 파일

- `v1_random_1000.png`: 랜덤 시작 궤적·시작 위치별 경로·mu-only 비교
- `v234_random_500.png`: v2-v4 새 랜덤 시작 direct-policy 궤적
- `results.json`: 전체 지표, 초기 y별 집계, 성공 및 경로 비율 Wilson 95% 구간
- 각 모델 폴더: `policy.npz`, `native.npz`, `provenance.json`, `verification.json`
- 초기 공식 모델 smoke의 직렬화 바이트 비교는 map 순서 때문에 실패했다. 이름별 모든 값·dtype·shape 일치 검사로 고친 후 별도 smoke를 통과했으며 실패 자료를 보존했다. 학습 모델 문제는 아니었다.
