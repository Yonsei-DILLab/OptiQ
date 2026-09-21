# MuJoCo: GPU당 두 run 동시 실행

사용자가 2026-09-20에 요청한 실행 병렬도 변경이다. 알고리즘, seed, 학습 길이,
평가 주기, precision, checkpoint와 W&B run ID는 변경하지 않는다.

- GPU0 Ant / GPU1 Humanoid / GPU2 HalfCheetah, 각각 최대2run.
- GPU3 GMM40와 CPU6–9는 그대로 둔다.
- 첫 slot CPU0–1/2–3/4–5, 두번째 slot CPU10–11/12–13/14–15.
- 현재 v1 seed0 training process를 종료하거나 checkpoint로 되돌리지 않는다.
- v1 기존 queue에는 STOP_NEW_RUNS만 써서 현재 자식이 끝난 후 추가 dispatch를 막는다.
- v2의 아직 대기 중인 queue worker만 종료한다. 학습 중이었다면 자동 변경을 중단한다.
- 새 dispatcher가 기존 training process를 발견하여 이미 차지한 slot으로 계산한다.
- 초기 병렬 실행 때 v1 seed1을 추가해 처리량을 확인했다. 이후 사용자 요청에 따라
  **대기 우선순위를 v2 seeds0,1,2,3 → v3 seeds0,1,2,3 → v1 seeds0,1,2,3으로 변경**한다. 실행 중인
  run은 그대로 두고, 완료/실패/이미 실행 중인 run을 제외한 다음 run을 고른다.
  순서는 우선순위이며 다른 환경이나 버전의 성공을 기다리는 dependency는 없다.
- 우선순위 변경 시 이전 dispatcher는 STOP_NEW_RUNS로 추가 dispatch만 막고
  살아 있게 둔다. 자식 training에 terminal hangup을 전파할 수 있는 parent/tmux
  종료를 피한다. 새 dispatcher가 active PID를 인계한다.
- v2와 v3 GPU 검증은 각각의 committed validator로 최초 실행 전에 수행한다.
- 기존 실패/중단 run은 자동 재시도하지 않으며 다른 independent seed는 eligible하다.

## 버전 및 저장

Numerical source는 immutable v1@9b26ade655097728e12aa63f8c274fe2350729f1 및
v2@947ad432be6719654feef20c94f09018faf7100e를 유지한다.
v3@46459857236aeac39d0c25a22485198f4a376200의12개 run을 추가해 총36개 계획이 된다.
v3는 동일한 z의 target-min explorer Q에서 target-max evaluator Q를 뺀 A>0일 때
raw A 가중 action MSE를 사용한다. 상세 수식은 해당 알고리즘 PROTOCOL.md에 있다.
최초 MC 평균 기반 v3 source(6b7c23e)는 사용자 정정에 따라 종료 및 비교 제외했다.
이 scheduler는 별도 commit으로
기록한다. 새로운 SCHEDULING.json에 numerical/source commit과 scheduler commit,
GPU/CPU/PID를 별도로 저장한다. 원래 W&B ID와 checkpoint provenance는 유지한다.

Source/queue는 legacy_monge/mujoco_two_per_gpu/COMMIT 아래 저장하며 기존 dildata
collector가 회수한다. 인증키는 원래의 승인된 별도0600 파일에서만 읽는다.

## 2026-09-20 W&B 시작 확인 실패 복구

v3 Ant seeds1,2는 `verify_online(run,0)`에서50초 제한을 넘겨 실패했다.
모델 생성 전이어서 replay/checkpoint/학습 step은 없었다. 이후 W&B 서버에는
확인 값이 올라온 것을 확인했다. Summary write의 가시성에만 의존한 짧은
health check를 학습 실패로 처리한 운영 오류다.

새 `online_runner.py`는 기존 immutable v3 source를 import하고 오직
`verify_online`만 교체한다. Probe를 `run.log(...,commit=True)`로 명시적으로
history에 전송하며 동일 token이 서버에 나타나는지 최대300초간 확인한다.
`Api.flush()`는 서버 업로드가 아니라 public API 캐시 초기화임을 구분한다.
Probe는 `ops/*` metric을 사용하여 `env_steps`를 조작하지 않는다.
알고리즘/optimizer/RNG/target/seed/원래 numerical SHA/W&B ID는 그대로다.
`LAUNCHER.json`과`ONLINE_HEALTH.jsonl`에 운영 코드 commit과 확인 결과를 기록한다.

`--retry-wandb-startup ant_v3_s1 ant_v3_s2`로 이번 두 startup 실패만 재등록한다.
실패 기록은 삭제하지 않고 `failure_history/`로 옮기며 기존 scheduler 기록도 보존한다.
다른 종류의 실패와 중단 run을 자동 재시작하지 않는다. 기존 활성 PID는 그대로
인계한다. GPU0당2run 제한을 유지하여 빈 slot이 생기면 실패했던 seed를 시작한다.
아직 시작하지 않은 v3도 새 logging wrapper를 사용한다. 이미 돌아가는 frozen
process에는 코드를 주입하거나 source 파일을 변경하지 않는다.

검증: mocked delayed visibility65초, 영구 미응답timeout, 일시적 API 오류,
실제 history commit/no env_steps 변경, 기존 queue 우선순위·중복 방지 검증.
별도의 `job_type=infrastructure-validation`, `exclude_from_results` W&B run으로
온라인 업로드도 확인한다. 이 검증은 MuJoCo 비교 seed에 포함하지 않는다.

## 처리량 비교

변경 전 약120초, 두 번째 run의 warmup 및 compilation이 끝난 뒤 약120초씩
progress.json과 GPU utilization을10초 간격으로 읽는다. 처리량은 environment
steps / 실제 경과시간이며 평가 및 checkpoint로 인한 중단을 포함한다.
새 run은5K uniform warmup을 제외하고 최소10K 이후 측정한다. 전후 구간이 짧고
학습 위치와 episode 길이가 다르므로 정밀한 알고리즘 speed benchmark로 해석하지 않는다.

VRAM 여유만으로2배속을 주장하지 않는다. 합산 처리량이 개선되는지를 보고 유지한다.
CAPACITY.json의 환경별 값을1로 내리면 추가 dispatch를 제한할 수 있다. 이미
돌아가는 run은 자동 kill하지 않는다. 감소가 필요한 경우 추가 run에SIGTERM을
보내 기존 full checkpoint 저장 경로로 종료시킨 후 재개 대상으로 기록한다.

## v4 extension, 2026-09-20

Add v4_heejoon_explorer source 1786b85c5a6d2c6250b6c8823714d09b2a151062:
v2 min–mean (mean of 16 target twin-min values) with raw exp(A) accepted-count
weighted regression. Ant/Humanoid/HalfCheetah, seeds 0–3, 1M steps each.
Preserve every active run. Dispatch preference is v2 → v3 → v4 → remaining v1.
Two runs/GPU, assigned environment GPUs 0/1/2, GPU3 unchanged. No environment
completion dependency. Include v4 in process adoption, GPU preflight, and repaired
W&B online wrapper. Existing v3 failure recovery is preserved, not repeated.
