# 기존 OptiQ dense reward 결과의 경로 다양성 검증

2026-09-24. 저장된 원시 rollout을 다시 집계했으며 신규 학습·평가 rollout은 실행하지 않았다. 각 실험 단일 seed 0.

**결론: v1의 이전 자체 포트 + NovelD 0.01 정책에는 두 성공 경로가 유지된다. 공식 환경 + NovelD OFF v1은 성공률이 높지만 거의 한 경로로 집중된다. 모든 maze에서 다양성을 유지한다고 말할 근거는 없다.**

## v1 최종 정책: 성공 경로와 실패 모두 집계

|실험|평가|+y 성공 경로|−y 성공 경로|실패|
|---|---|---:|---:|---:|
|이전 포트 dense + NovelD 0.01, 1M|policy-natural|44|55|1|
|이전 포트 dense + NovelD 0.01, 1M|policy-fixed|71|29|0|
|이전 포트 dense + NovelD 0.01, 1M|native-fixed|75|21|4|
|이전 포트 dense + NovelD 0.01, 1M|zero_z-fixed|100|0|0|
|공식 환경 dense + NovelD OFF, 3.008M|policy-natural|1|92|7|
|공식 환경 dense + NovelD OFF, 3.008M|policy-fixed|0|98|2|
|공식 환경 dense + NovelD OFF, 3.008M|native-fixed|0|85|15|
|공식 환경 dense + NovelD OFF, 3.008M|zero_z-fixed|0|0|100|

policy는 fresh random z + conditional sigma, native는 random z의 mu-only, zero_z는 z=0 mu-only다. 평가에 외부 DACER 잡음이나 NovelD 보상을 더하지 않았다.
natural은 v1 랜덤 시작. fixed는 각 실험 내부에서 qpos/qvel 및 저장된 초기 시뮬레이터 상태를 동일하게 복원했다. 두 실험의 fixed 시작은 서로 다르다: 이전 포트 (0,0), 공식 환경 약 (1.9873,−1.4523). 그림은 +y가 위로 향하도록 통일했다.
이전 포트 v1은 500k부터 1M까지 21/21회의 평가에서 두 성공 경로가 모두 관찰됐다(각 10회). 한 번만 우연히 나타난 결과가 아니다.
이전 포트 v1의 고정 시작 mu-only에서도 75/21로 나뉘고 z=0에서는 100/0이다. 시작점 차이나 conditional sigma 잡음만으로 설명되는 현상이 아니며, latent를 사용하는 정책의 경로 다양성에 대한 근거다. 다만 동일 state의 action density 자체가 다봉이라는 직접 증명은 아니다.
별도 과거 모델 복원 감사에서도 새 평가 난수로 v1 policy 100회를 실행해 +y 71, −y 28, 실패1을 얻었다. 현재 감사에서는 그 저장 결과와 model-unchanged 검증서를 확인했다.

## 공식 환경 dense + NovelD OFF의 미로별 마지막 저장 평가

|미로|저장 평가 step|상태|성공/평가수|성공 경로 집계|
|---|---:|---|---:|---|
|v1|3,008,256|완료|93/100|G1/lower: 92, G1/upper: 1|
|v2|2,250,240|사용자 중단 전 마지막 평가|18/20|goal(8,0)/central-corridor: 18|
|v3|3,500,032|사용자 중단 전 마지막 평가|8/20|G1/passage-y+4: 8|
|v4|2,250,240|사용자 중단 전 마지막 평가|0/20|성공 없음|

v2-v4는 완료 결과가 아니고 마지막 평가는 20회뿐이다. v4는 성공 경로가 없어 모드 붕괴를 판정할 수 없다. 한 목표에 도달했다는 사실과 한 경로를 사용했다는 사실은 구분하고 실제 통로 통과 좌표로 분류했다.

## 이전 포트 dense + NovelD 0.01: 1M 최종 고정 상태 direct-policy

|미로|성공/100|성공 경로 집계|
|---|---:|---|
|v1|100|G1/upper: 71, G1/lower: 29|
|v2|100|goal(8,0)/central-corridor: 100|
|v3|99|G1/passage-y+4: 99|
|v4|99|G1/upper-entry/upper-outer: 99|

v2의 goal ID는 과거 포트와 공식 환경에서 반대로 부여되므로 실제 좌표로 비교했다. v2-v4 이전 포트는 natural과 fixed의 시작 상태 분포가 같으므로 200개 독립 시작 상태처럼 합치지 않는다.

## 모델과 실험 조건

|항목|이전 포트 dense + 0.01|공식 환경 dense OFF v1|현재 실행 중 dense OFF|
|---|---|---|---|
|actor/critic|256×2|256×2|256×3|
|temperature|0.25|0.25|0.01|
|LR actor / critic|3e-4 / 3e-4|3e-4 / 3e-4|3e-4 / 5e-4|
|환경 수 / batch|1 / 256|256 / 4096|256 / 4096|
|업데이트 비율|1 / transition|8 / 256 transitions|8 / 256 transitions|
|최종 transitions / learner updates|1M / 995,000|3,008,256 / 93,752|v1 예정 3,008,256 / 93,752|
|NovelD|0.01|OFF, RND updates 0|OFF|

모든 세대의 OptiQ는 여기서 random z, mean-init1, log sigma[-5,-1], DACER behavior exploration을 사용한다. 본 비교의 차이는 NovelD 하나가 아니므로 ON/OFF 인과효과나 현재 설정의 최종 성능을 단정하지 않는다.
이전 포트 v1-v4 full state를 CPU에서 읽어 원본 SHA256, actor/critic 실제 256×2 kernel, optimizer/model 유한성을 검증했다. 공식 환경 v1의 307MB final checkpoint도 서버 CPU에서 SHA256·93,752 updates·RND OFF·실제 256×2·유한한 정책/optimizer를 확인했다. 공식 환경 v2-v4는 final-only 저장 설정으로 중단 당시 full checkpoint가 없으며 원시 평가 궤적을 사용했다.
저장 궤적 전부에서 유한한 좌표, 실제 goal 반경 도달, dense return 거리합, 고정 상태 동일성을 다시 검사했다. 입력 SHA256은 results.json에 기록했다.

## 출처와 파일

- 이전 포트 학습 source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`
- 공식 환경 dense OFF 학습 source: `26336810f7ea4ca61210ea70c6aeeae9f7acaed0`
- 현재 실행 중 dense OFF source: `a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3` (이번 과거 분석의 결과에 섞지 않음)
- 이번 재집계: `analyze.py`, `results.json`, `legacy-model-verification.json`, `upstream-model-verification.json`
- 이전 새 난수 모델 복원 확인: `../antmaze_route_audit/rerollout-verification.json`
- 그림: `v1_policy_comparison.png`, `v1_route_persistence.png`, `upstream_dense_off_latest.png`

현재 16개 실험은 이 분석을 위해 설정을 바꾸거나 중단하지 않았다.
