# v5 실험과 재현 설정

현재 코드 기본값은 T=.25, **Sinkhorn epsilon=.1, actor·critic gradient clip 없음**이다.
v5 기본 profile에 이 값을 명시했고 optional v5 exploration profile도 상속한다.
v2/v3/v4 profile의 기본값은 바꾸지 않았다.

## 1. 현재 Ant grid

환경 Ant-v4, seed 0·1·2·3, 조합별 4개, 각각 1M environment step이다.
모든 시드에 같은 T와 Sinkhorn epsilon을 적용하며 seed별 자동 선택은 없다.

| 실행 순서 | Sinkhorn epsilon | Teacher T | 시드 | 처리 |
|---|---:|---:|---|---|
| 기존 cohort | .1 | .25 | 0·1·2·3 | 이미 등록된 no-clip 4개 재사용 |
| 후속 1 | .1 | .1 | 0·1·2·3 | fresh 4개 |
| 후속 2 | .1 | .05 | 0·1·2·3 | fresh 4개 |
| 후속 3 | .05 | .25 | 0·1·2·3 | fresh 4개; 이 조합도 포함 |
| 후속 4 | .05 | .1 | 0·1·2·3 | fresh 4개 |
| 후속 5 | .05 | .05 | 0·1·2·3 | fresh 4개 |

총 24개 = 기존 4개 + 후속 20개다. 기존 T=.25는 **epsilon=.1 조합만**
재사용한다. Epsilon=.05 / T=.25를 제외하지 않는다. 기존 clip=2인 실행을
no-clip 결과로 재사용한 것도 아니다.

각 cohort의 네 시드를 GPU 0·1·2·3에서 함께 실행하고, 네 개 모두 정상적으로
1M을 끝낸 뒤 다음 cohort를 시작한다. 낮은 점수나 급락으로 자동 중단하지
않는다. 실행 오류는 완료로 표시하지 않으며 자동 재시도·checkpoint resume은 없다.

## 2. 공통 설정과 실제 차이

| 항목 | 값 |
|---|---|
| Profile / learner | `mujoco_v5`, mean-action OT |
| Actor / critic | 256x2 / 256x2, GELU, LayerNorm·BatchNorm 없음 |
| 초기 sigma / mean head scale | .5 / 1e-4; smallinit 아님 |
| Teacher | conditional Gaussian mixture, M=16, K=64, exact sampling, anchor 없음 |
| Importance weights | beta=1, adaptive beta 없음 |
| NLL | full OT row, pre-tanh Gaussian, mu·sigma 둘 다 학습 |
| Sinkhorn | epsilon=.1 또는 .05, **100회**, raw squared action cost |
| Optimizer | actor·critic Adam LR=3e-4, **gradient clip 없음** |
| Backup | 일반 TD, target twin-min; policy entropy bonus·soft guard 없음 |
| Uniform / annealing | 추가 uniform 0%, temperature schedule 없음 |
| Warmup / update | 5K random collection, 이후 actor·critic 각각 매 step 1회 |
| 평가 | zero-z / sampled-z, 둘 다 Gaussian epsilon=0, 5K마다 각각 10 episode |
| Checkpoint | 5001 및 이후 50K마다 actor·critic state |
| 총량 | run별 1M env steps, 995K actor·critic updates |
| W&B | [OptiQ/v5-jaehoon](https://wandb.ai/OptiQ/v5-jaehoon) |

기존 no-clip cohort 대비 학습 설정 차이는 **teacher T와 Sinkhorn epsilon**
두 필드뿐이다. 명령·resolved config를 모든 시드에서 대조했다. 학습 소스는
`71c5ba86cbf63042b227f37cce114143700378c9`의 frozen clone이며 각 run에
`alg.optimizer.ac_grad_norm=null`과 epsilon을 명시했다. 새 repository 기본값은
이 기존 설정을 기본 profile로 옮긴 것이므로 frozen source나 worker를 교체하지 않는다.

## 3. 실행 상태와 기록 위치

상태는 [24개 run snapshot](experiments/ant_sinkhorn_temperature_noclip_20260913.json)의
`snapshot_utc`, `status_at_snapshot`, `wandb_url`로 확인한다. 문서 작성 시
epsilon=.1 / T=.25 네 개는 완료됐고, epsilon=.1 / T=.1 네 개가 실행 중이며
나머지 16개는 대기 중이었다. 이는 날짜가 붙은 기록이고 실시간 상태는 아니다.
대기 run의 W&B ID는 실제 worker 시작 시 새로 생성된다.

| 구분 | Campaign directory / service |
|---|---|
| 기존 no-clip 4개 | `/root/optiq-experiments/ant_v5_meanOT_sinkhorn010_noclip_20260913T1420` |
| 후속 20개 | `/root/optiq-experiments/ant_v5_sinkhorn_temperature_noclip_20260913T1633` |
| 현재 dispatcher | `optiq-ant-v5-eps-temp-noclip-20260913` |

Campaign의 `manifest.json`은 명령, resolved config, source/dependency hashes를
보관한다. `state.json`은 각 job의 실행 상태, `controller_state.json`은 현재
단계다. `registration_verification.json`은 등록 당시 검사 기록이다.
각 output의 `config.json`, `progress.csv`, `completed.json`, checkpoint와
`evaluations_zero_z.npz` / `evaluations_stochastic_z.npz`가 실제 run 기록이다.
전체 1M 완료 시 두 평가 파일은 각각 201개 시점 x 10개 episode를 가진다.

## 4. 설정 검사와 재현 명령

다음은 새 기본값 검사다. `--check`는 학습이나 W&B run을 만들지 않는다.

```bash
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-jaehoon

# 예: T=.25, Sinkhorn epsilon=.05. No-clip은 기본값에서 상속한다.
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
bash scripts/run_v5.sh 3 --check benchmark=ant wandb.project=v5-jaehoon \
  alg.actor.temperature=.25 alg.actor.sinkhorn_epsilon=.05

# 기존 clip=2, epsilon=.25 mean OT recipe의 설정 재현.
OPTIQ_PYTHON=/root/.venv-optiq-mujoco/bin/python \
bash scripts/run_v5.sh 0 --check benchmark=ant wandb.project=v5-test \
  alg.actor.sinkhorn_epsilon=.25 alg.optimizer.ac_grad_norm=2.0
```

학습은 승인된 job을 supervisor에서 관리하며 launcher의 `--check`를 제거한다.
이미 등록된 위 grid를 재생성하거나 이 명령으로 중복 실행하지 않는다.
이전 commit에서도 현재 no-clip recipe를 재현하려면
`alg.actor.sinkhorn_epsilon=.1 alg.optimizer.ac_grad_norm=null`을 명시한다.
비교에는 seed·dependency versions·평가 방법도 함께 맞춰야 한다.

W&B YAML 기본값은 역사적 `v4-test`를 유지한다. 현재 grid의 모든 command에
`wandb.entity=OptiQ wandb.project=v5-jaehoon`을 명시했고, 과거 탐색 결과는
`v5-test`에 보존한다. Group과 run name에 epsilon, T, no-clip, seed를 표기한다.

## 5. 과거 기록과 혼동하지 않을 것

| 기록 | Sinkhorn epsilon | Gradient clip | Project / 해석 |
|---|---:|---:|---|
| 최초 완료 mean OT 대조 4개 | .25 | 2.0 | v5-test로 기록 복사; 원본 checkpoint 보존 |
| 완료 uniform / anneal / 둘 다, 12개 | .25 | 2.0 | v5-test, [결과와 한계](RESIDUAL_DIAGNOSIS_KO.md) |
| 이후 epsilon=.1의 clipped 4개 | .1 | 2.0 | v5-test; 사용자 요청으로 중단, 완료로 취급하지 않음 |
| 현재 24개 grid | .1 / .05 | **없음** | v5-jaehoon; 해당 epsilon/T끼리 비교 |

최초 mean OT 대조의 후반 보상 5,265(sampled-z)는 epsilon=.25 / clip=2 조건의
결과다. 새 기본값의 보상으로 인용하지 않는다. 기존 수치 진단도 당시 저장
모델을 사용했으므로 no-clip이나 epsilon 축소의 전체 학습 효과를 증명하지 않는다.

`configs/v5/final.yaml`의 기본값 변경은 기존 탐색 profile에도 상속되지만,
완료 탐색 실험의 소스·명령·결과는 고정되어 있다. 과거 설정을 재현하려면
[탐색 문서](EXPLORATION.md)의 명시적 override를 사용한다.
