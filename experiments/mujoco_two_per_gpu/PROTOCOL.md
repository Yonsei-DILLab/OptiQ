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
- 첫 추가 run은 각 환경 v1 seed1이다. 같은 algorithm 설정에서 1개/2개 합산
  처리량을 비교하기 위함이다. 이후 order는 v2 seeds0,1 → v1 seeds2,3 → v2 seeds2,3.
  순서는 우선순위이며 다른 환경이나 버전의 성공을 기다리는 dependency는 없다.
- v2 GPU 검증은 기존 committed validator로 최초 실행 전에 수행한다.
- 기존 실패/중단 run은 자동 재시도하지 않으며 다른 independent seed는 eligible하다.

## 버전 및 저장

Numerical source는 immutable v1@9b26ade655097728e12aa63f8c274fe2350729f1 및
v2@947ad432be6719654feef20c94f09018faf7100e를 유지한다. 이 scheduler만 새 commit으로
기록한다. 새로운 SCHEDULING.json에 numerical/source commit과 scheduler commit,
GPU/CPU/PID를 별도로 저장한다. 원래 W&B ID와 checkpoint provenance는 유지한다.

Source/queue는 legacy_monge/mujoco_two_per_gpu/COMMIT 아래 저장하며 기존 dildata
collector가 회수한다. 인증키는 원래의 승인된 별도0600 파일에서만 읽는다.

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
