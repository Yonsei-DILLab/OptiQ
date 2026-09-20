# v8 구현 검증 기록

2026-09-20, Python 3.11 / JAX 0.4.33 / RTX 4090 환경에서 검증했다.
이 문서는 코드·연결·재개 검증 기록이며 GMM40 분포 복원 성공을 뜻하지 않는다.

## 자동 검증

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/test_v8_actor.py tests/test_v8_transport.py \
  tests/test_v8_rl.py tests/test_v7_config.py tests/test_v7_rl.py \
  tests/test_v7_actor.py -q --disable-warnings
```

103개 통과. 추가 `tests/test_v8_gmm.py` 6개 통과: 총 109개.

- OT를 batch로 푼 결과와 상태마다 따로 푼 결과, teacher 위치의 연속 배정과 P의 일치.
- 새 행동의 assignment gradient와 수치 미분 일치, 이전 actor/teacher/map의 gradient 차단.
- Proposal sigma floor가 actual source Gaussian 비용에 들어가지 않는지 확인.
- Importance correction 한 번 적용, source 비중 재보정 없음, teacher 비유한 입력 거부.
- OT 미수렴 시 actor·Adam·RNG 보존. Chunk에서 첫 실패 즉시 중단, 성공한 prefix 보존.
- No-clip 및 clip=2 설정 각각에서 checkpoint 이후 동일 trajectory로 정확히 재개.
- Ant 6 environment steps에서 H=4096, M=256→K=16으로 replay→soft TD→actor 연결 및 저장·복원.
- 기존 v7 설정과 학습 경로의 회귀 검증.

## GPU full-shape preflight

Batch=256, H=4096, M=256→K=16, alpha=1, MLP=256×2,
Adam=3e-4, clipping 없음, Sinkhorn min/max=10/500, 상대오차 한계=1e-3.

실제 7회 업데이트, actor·Adam·RNG 정확한 재개, 평가 RNG 격리를 확인했다.
초기 OT는 10회 반복으로 수렴했고 관측한 최대 행 상대오차는 8.94e-7,
열 상대오차는 7.15e-7이었다. GPU 0–3 모두 정상 조회되었다.
별도 GPU 1에서 실제 runner의 2-step 학습·평가·저장을 완료하고,
그 checkpoint로 새 디렉터리에서 step 3까지 재개하는 경로도 통과했다.

초기 JIT compilation을 제외한 **시작 구간 4회 업데이트**의 중앙 속도는
9.43 ms/update였다. 이 숫자는 장기 처리량 추정치가 아니다. 학습 후 Gaussian이
분리되면 OT 반복 횟수와 시간이 증가할 수 있으므로 실제 `runtime.json`과
`training.jsonl`의 OT residual/iterations를 함께 확인한다.

검증 산출물은 저장소 밖 `/root/optiq-experiments/v8/validation/`에 보관한다.
학습 runner는 preflight에 기록된 numerical source SHA256과 현재 소스가 다르면
시작을 거부한다. 체크포인트도 설정 signature, 전체 optimizer 및 RNG 상태를 검사한다.

## 범위

GMM의 고정 Q에 대한 100K 분포 복원 실험과 RL 장기 성능은 이 검증과 별개다.
실험은 commit된 소스를 고정해 실행하며, 학습곡선과 mode coverage를 보고 판단한다.
외부 baseline 코드는 기존 구현을 보존했으며 v8 검증을 이유로 다시 학습하지 않았다.
