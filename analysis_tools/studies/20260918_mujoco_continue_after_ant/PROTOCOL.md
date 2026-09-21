# login4 MuJoCo: interrupted Ant를 보류하고 Humanoid부터 계속

2026-09-18 사용자 지시: 중단된 login4 Ant는 그대로 두고 다음 작업들을 진행한다.
이 운영 변경을 commit/push한 뒤 기존 Humanoid array의 Ant dependency만 해제한다.

| Temperature | 원래 source/launch commit | Ant array (수정 없음) | Humanoid | Hopper | Walker2d | HalfCheetah |
|---|---|---|---|---|---|---|
| 0.5 | ec9bb32b424a37147ea6f3dacf08e9315f28a622 | 2277773 | 2277774 | 2277775 | 2277776 | 2277777 |
| 0.25 | 23f226e0da65cf493afe27b9087a364d534a81cd | 2277872 | 2277873 | 2277874 | 2277875 | 2277876 |

모든 환경은 기존 Direct GMM N128×M256, seed0–3, 각1M steps 그대로다.
Humanoid → Hopper → Walker2d → HalfCheetah의 afterok 체인은 유지한다.
기존 job IDs, frozen numerical source, config, W&B group, backup은 변경하지 않는다.
두 temperature는 독립적으로 진행되며 동시에 최대8 GPU(각4)를 사용한다.

Ant T0.5 seed3은658,437 steps, T0.25 seed2는929,958 steps에서 SLURM
preemption으로 중단됐다. 이들은 미완료로 남겨두고 재실행·대체·완료 처리하지 않는다.
기존 `status/ant_*.json`의 running 값은 강제 종료 후 갱신되지 않은 기록일 수 있으므로
분석에서는 SLURM 종료 로그 및 이번 운영 기록을 함께 확인한다.

실행 명령은 다음과 같다.

```sh
scontrol update JobId=2277774 Dependency=
scontrol update JobId=2277873 Dependency=
```

실행 스크립트는 source hash와 validation 결과, 모든 후속 작업의 원래 dependency를
검사한다. 변경 전후 scheduler 기록과 이 운영 변경 commit을 각 캠페인의
`CONTINUATION_20260918.json`에 저장한다. 원래 source commit은 그대로 유지한다.
기존 protocol의 “모든 Ant seed 성공 후 진행”에 대한 사용자 승인 예외이며,
재학습 또는 새로운 비교 설정이 아니다. 기존 protocol에서 big_qos를 비선점으로
표현했지만 실제 두 종료 로그에는 PREEMPTION이 기록되어 있어 보호를 보장하지 않는다.

재개 후 등록 상태와 초기 실행 로그를 확인한다. 전체1M 완료를 기다리지 않는다.
