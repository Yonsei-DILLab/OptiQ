# OptiQ v5 문서

현재 v5는 **mean-action OT + full conditional Gaussian NLL + 일반 TD**다.
2026-09-13 사용자 지정 기본값은 **T=.25, Sinkhorn epsilon=.1 / 100회,
actor·critic gradient clip 없음**이다. 네트워크는 256x2, 초기 sigma=.5,
추가 uniform·annealing은 꺼져 있다.

| 문서 | 포함 내용 |
|---|---|
| [전체 의사코드](PSEUDOCODE.md) | 설정·shape, 행동 생성, TD, proposal 밀도, 중요도 가중치, Sinkhorn, NLL, 전체 학습 루프, 두 평가 |
| [구현 세부 사항](IMPLEMENTATION_DETAILS_KO.md) | tanh를 쓰는 이유, 연속 mixture와 유한 proposal, sigma의 역할, stop-gradient, NLL 미분, 수치 안정성, 지표 해석 |
| [v4에서 달라진 점](CHANGES_KO.md) | 최초 mean OT 변경과 이후 epsilon/no-clip 기본값 변경, 과거 대조의 범위 |
| [실험·재현 설정](EXPERIMENTS_KO.md) | 2x3x4 Ant grid, W&B, 현재와 과거 설정, CLI, 실행 순서, 기록 파일 |
| [선택적 탐색](EXPLORATION.md) | uniform 10%, 40K annealing, 과거 실행과 현재 profile 상속 관계 |
| [수집 best-of-8](BEST_OF_8.md) | Kb=8 / Kt=1, 학습 행동 수집만 twin-min 선택, 기존 두 평가 유지 |
| [구현 검증](VALIDATION.md) | 기존 수치 parity와 새 기본값 회귀·통합 검사 |
| [탐색 profile 검증](EXPLORATION_VALIDATION.md) | collection hook, schedule, 평가 RNG 분리 |
| [잔여 학습 지연 진단](RESIDUAL_DIAGNOSIS_KO.md) | sigma 관련 확인 결과, frozen-model 수치 실험, 반증된 설명, 아직 식별하지 못한 전체 원인 |

## 설정을 읽는 기준

`configs/v5/final.yaml`이 현재 기본값이며 `mujoco_v5`는 alias다.
Default benchmark는 Humanoid-v4이므로 Ant는 `benchmark=ant`로 선택한다.
현재 Ant 기록 목적지는 `wandb.project=v5-jaehoon`을 명시한다. YAML의
역사적 W&B 기본값 `v4-test`를 현재 실험 목적지로 오해하지 않는다.

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-jaehoon
```

이 명령은 설정만 검사한다. 새 기본값이 과거 frozen source를 바꾸지는 않는다.
현재 Ant grid는 commit `71c5ba8`에 epsilon/no-clip override를 명시한 상태로
실행되며, 새 기본 profile과 학습 설정을 대조해 기록했다.

## 용어

| 표기 | 의미 |
|---|---|
| T | `softmax(Q/T - log q)`의 teacher 온도 |
| Sinkhorn epsilon / lambda_OT | 수송 plan의 entropic 정규화 계수; 기본 .1 |
| Gaussian epsilon | `mu + sigma * epsilon`의 표준정규 난수; 위 계수와 무관 |
| z | 행동마다/학습 state마다 샘플링하는 연속 latent |
| Mean OT | 학생 위치를 `tanh(mu(s,z))`로 사용; teacher와 NLL에는 sigma 유지 |
| No clip | actor·critic의 **gradient clipping 없음**; tanh·log-std 범위·twin-min TD는 유지 |
| 두 평가 | `z=0, epsilon=0` 및 `z~Normal, epsilon=0`; 각각 별도 기록 |

Mean OT 대조가 개선된 것과 남은 학습 지연의 원인을 증명한 것은 별개다.
기존 수치 진단은 epsilon=.25 / clip=2 모델을 대상으로 했으며, 현재 grid의
미완료 결과를 결론에 섞지 않는다.
