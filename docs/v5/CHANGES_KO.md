# v4 → v5 변경점

v5는 완료된 Ant-v4 mean-action OT 대조를 정식 실행 profile로 옮긴 버전이다.
v4 branch `558ad09`에서 분기했으며, 실험의 보존 source `6f5c987`에 적용했던
OT student 입력 변경을 `alg.actor.ot_student_action=mean`으로 선택한다.

## 실제 알고리즘 차이

| 단계 | v4 기본 | v5 기본 |
|---|---|---|
| OT student 위치 | `tanh(mu + sigma*epsilon)` | **`tanh(mu)`** |
| Teacher 후보 | Conditional Gaussian mixture, sigma 사용 | 동일 |
| Proposal density 및 beta=1 중요도 보정 | Sigma 포함 | 동일 |
| OT row / column marginal | 1/16 / teacher weights | 동일 |
| OT 정규화·반복 | .25 / 100 | 동일 |
| Full-row conditional Gaussian NLL | Mu와 sigma 학습 | 동일 |
| 환경 행동·TD next action | Gaussian noise와 latent 모두 샘플링 | 동일 |
| Critic backup | Plain TD, target twin min-Q | 동일 |
| Policy entropy / guard | 없음 | 동일 |
| 평가 | zero-z / sampled-z, 두 모드 epsilon=0 | 동일 |
| Actor / critic | 256×2 / 256×2 | 동일 |
| 초기 sigma / mu head scale | .5 / 1e-4 | 동일 |
| 추가 uniform / annealing | 기본 profile에는 없음 | 없음 |

Student epsilon draw를 유지하므로 비용 입력을 바꿨다는 이유로 이후 RNG key의
순서를 당기지 않는다. 이 draw는 기존 sampled-action 진단에도 계속 쓰인다.
Teacher, OT plan, NLL target은 stop-gradient이며 별도 Q-gradient loss를 더하지 않는다.

## 실행 설정 차이

| 설정 | v4 기본 profile | v5 기본 profile |
|---|---|---|
| Entry / alias | `mujoco_v4` / `v4/final` | `mujoco_v5` / `v5/final` |
| Teacher T | .1 | **.25 고정** |
| OT 위치 선택 설정 | 생략하면 `sample` | `ot_student_action: mean` |
| W&B project | 기본 `v4_test`, 기존 실험은 사용자 지정 `v4-test` | 사용자 지정 `v4-test` 유지 |
| W&B group / job type | v4 | v5_meanOT / v5-comparison |
| 출력 위치 | `optiq-experiments/v4/outputs` | `optiq-experiments/v5/outputs` |
| Launcher | `scripts/run_v4.sh` | `scripts/run_v5.sh` |

**T=.25는 검증한 실험 조건을 기본값으로 채택한 설정 변경이며, mean-action OT의
수학적 필수 조건이 아니다.** 완료된 인과 대조는 양쪽 모두 T=.25였으므로 그 결과에
온도 변경 효과가 섞여 있지 않다. 필요하면 v5 launcher에서 다른 고정 T를 지정할 수 있다.
Benchmark 기본값은 상속한 Humanoid-v4를 유지하며 Ant는 `benchmark=ant`로 지정한다.

명시적인 v2/v3/v4 config는 기존 sample-action OT를 유지한다. v5 launcher는
mean-action OT, Gaussian teacher/NLL, plain TD, 두 평가, 추가 uniform 없음,
annealing 없음이라는 계약을 검사한다. Supervisor template은 제공하되 설치·시작은
별도 실험 실행 요청에 따른다.

이후 사용자가 요청한 추가 탐색 실험은 [별도 profile](EXPLORATION.md)로 제공한다.
`mujoco_v5_behavior010`, `mujoco_v5_annealing`, `mujoco_v5_annealing_behavior010`은
각각 uniform 10%, 기존 40K annealing, 두 옵션의 결합이며 기본 v5 설정은 유지한다.

## 왜 이 입력을 바꾸는가

기존 OT는 같은 `(s,z)`라도 매번 다른 student epsilon에 따라 teacher 배정이 달라진다.
반면 NLL이 학습하는 mu와 sigma는 `(s,z)`만의 함수다. Epsilon에 의존한 배정 차이는
일관된 z별 평균 역할로 학습되기보다 sigma의 잔차 분산에 흡수될 수 있다.
v5는 student 위치의 epsilon 의존성을 제거한다. Teacher sampling의 무작위성은 유지한다.

Sigma를 포기하는 변경은 아니다. Teacher 생성·density·NLL에 sigma가 남아 있고,
sigma head는 각 z에 배정된 teacher 샘플의 잔차 분산을 학습한다. 다만 배정 비용이
student의 sigma 자체를 구분하지 않으므로, mu가 같고 sigma만 다른 두 component는
같은 OT 비용 행을 가진다. Mean-only OT가 모든 mixture 분해를 보존하지는 않는다.

## 관측된 효과와 한계

Ant-v4, T=.25, seed 0·1·2·3, 1M step, 추가 uniform/annealing 없는 대조:

| 900K–1M 평균 | 기존 v4 | Mean-action OT |
|---|---:|---:|
| zero-z, epsilon=0 | 4,458 | **5,283 (+18.5%)** |
| sampled-z, epsilon=0 | 4,420 | **5,265 (+19.1%)** |

기존 약 3,800점에 머문 시드 1·3이 약 5,300–5,400점으로 개선됐다. 공통 상태와
z를 넣은 50K actor에서 mean-action spread가 14.6배, mu 절대 분산이 약 222배였다.
그러나 seed 0·2의 초중반은 더 느렸고, 1M에는 z별 평균 차이가 다시 작아졌다.
초기 분화가 성능 개선을 유일하게 매개했는지, v1과의 모든 격차를 해소했는지,
다른 환경에서도 개선되는지까지 증명한 결과는 아니다.

이 결과는 새 v5 branch를 1M 재학습한 결과가 아니라 **v5에 옮긴 개입의 기존
완료 대조 결과**다. 이전 W&B run을 v5로 이름 변경하거나 재사용하지 않았다.
[시드별 곡선·원자료·해석](../v4/MEAN_OT_RESULTS_KO.md)과 [구현 검증](VALIDATION.md)을 구분해서 읽는다.

## 코드 대응

| 역할 | 파일 |
|---|---|
| 기본 profile | [configs/v5/final.yaml](../../configs/v5/final.yaml) |
| 비용 위치 선택 / config 전달 | [algorithm.py](../../optiq_dime/algorithm.py), `train → _train → update_actor` |
| 공통 config 검증 / 기본 entry | [run_optiq_dime.py](../../run_optiq_dime.py) |
| v5 profile 검사 | [verify_v5.py](../../scripts/verify_v5.py) |
| 학습 launcher | [run_v5.sh](../../scripts/run_v5.sh) |
| 의사코드 | [PSEUDOCODE.md](PSEUDOCODE.md) |
