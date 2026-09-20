# v8 구현 검증 기록

2026-09-20, Python 3.11 / JAX 0.4.33 / RTX 4090 환경에서 검증했다.
현재 대상은 **raw z → teacher 거리 OT** 구현이다. 이전 Gaussian-cost 검증은 아래
별도 기록으로 구분한다. 이 문서는 코드·연결·재개 검증이며 분포 복원 성공을 뜻하지 않는다.

## 자동 검증

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/test_v8_actor.py tests/test_v8_transport.py \
  tests/test_v8_rl.py tests/test_v7_config.py tests/test_v7_rl.py \
  tests/test_v7_actor.py tests/test_v8_gmm.py -q --disable-warnings
```

Raw-z 변경 후 **112개 통과**: RL/v7 회귀 95개, actor/transport 11개, GMM 6개.

- OT를 batch로 푼 결과와 상태마다 따로 푼 결과, teacher 위치의 연속 배정과 P의 일치.
- 새 행동의 assignment gradient와 수치 미분 일치, 이전 actor/teacher/map의 gradient 차단.
- 고정 teacher에서 actor sigma와 alpha를 바꿔도 raw-z 비용과 배정이 변하지 않음.
- Actor를 H개 source에서 평가하지 않고, teacher 후보와 선택한 latent에서만 평가함.
- Importance correction 한 번 적용, source 비중 재보정 없음, teacher 비유한 입력 거부.
- OT 미수렴 시 actor·Adam·RNG 보존. Chunk에서 첫 실패 즉시 중단, 성공한 prefix 보존.
- No-clip 및 clip=2 설정 각각에서 checkpoint 이후 동일 trajectory로 정확히 재개.
- Ant 6 environment steps에서 H=4096, M=256→K=16으로 replay→soft TD→actor 연결 및 저장·복원.
- 기존 v7 설정과 학습 경로의 회귀 검증.

## GPU full-shape preflight

Batch=256, H=4096, M=256→K=16, alpha=1, 거리 epsilon=.1, MLP=256×2,
Adam=3e-4, clipping 없음, Sinkhorn min/max=10/2000, 상대오차 한계=1e-3.

수정 후 full-shape 검증은 **300회 업데이트 + 정확한 재개 1회**로 강화했다.
Actor·Adam·RNG 정확한 재개와 평가 RNG 격리를 확인했다.
마지막 100회 구간의 최대 행 상대오차는 0.000999987,
열 상대오차는 2.21e-6이었다. GPU 0–3 모두 정상 조회되었다.
GPU preflight는 실제 비용이 raw z 제곱거리인지, H Gaussian forward가 없는지도 검사한다.

초기 JIT compilation을 제외한 **100-update block 두 개**의 중앙 속도는
62.42 ms/update였다. 이 숫자는 장기 처리량 추정치가 아니다.
마지막 block의 lane 평균 OT 반복 횟수는 87.32, 최대는 228회였다.
H actor forward 제거가 전체 학습 속도 개선을 보장하는 것은 아니다. 매 상태의
fresh OT를 엄격한 오차까지 푸는 시간이 필요하므로 실제 `runtime.json`과
`training.jsonl`의 warmed 시간·반복 횟수를 함께 확인한다.

검증 산출물은 저장소 밖 `/root/optiq-experiments/v8/validation/`에 보관한다.
학습 runner는 preflight에 기록된 numerical source SHA256과 현재 소스가 다르면
시작을 거부한다. 체크포인트도 설정 signature, 전체 optimizer 및 RNG 상태를 검사한다.

## 이전 Gaussian-cost prototype에서 발견한 수렴 판정 오류

첫 실행은 seed 0–3에서 85/94/96/87회 승인 후 수치 검사로 중단됐다.
내부 반복은 `exp(kernel+f+g)`로 종료를 판정했지만 최종 P는
`teacher_weight * softmax(kernel+f)`로 반환했다. 수학적으로 같은 식이어도
float32 반올림 때문에 최종 행 오차가 0.001을 약 1.6e-7–3.4e-7 초과할 수 있었다.

내부 종료와 최종 검사가 **동일한 normalized-plan 함수**를 사용하도록 수정했다.
허용오차 1e-3은 그대로다. 간단한 2×2 kernel에서 이전 구현의 조기 종료를 재현하는
회귀 시험을 추가했다. 실패한 실험의 소스·로그·체크포인트는 그대로 보존했다.
실패 chunk에서 실제로 소요한 시간도 `runtime.json`에 포함하도록 수정했다.

별도 continuation 진단에서 330번째 업데이트는 실제로 500회 반복 안에 수렴하지 않았다.
같은 actor/teacher/RNG에서 702회에 수렴했으므로 solver 상한만 2000으로 늘렸다.
당시 목표 비용·epsilon·marginal 허용오차는 그대로였다. 이 설정은 full-shape 301-update
preflight와 정확한 재개 검증을 다시 통과했다.
별도 상태 이식 진단은 429회까지 성공했으나 최대 1995회를 사용했다.
그 진단은 임시 logger 오류로 멈췄으며 solver 실패로 해석하지 않는다.
당시 사용자가 비용 설계 검토를 요청해 추가 연속학습을 보류했다.
이후 raw-z 거리 비용을 선택해 새 코드와 새 디렉터리에서 100K 실험을 시작했다.
이전 prototype의 체크포인트를 raw-z 실험으로 재개하지 않았다.
현재 raw-z의 2000회 한도 역시 100K 전 구간에서 충분한지는 계속 관측해야 한다.

## 범위

GMM의 고정 Q에 대한 100K 분포 복원 실험과 RL 장기 성능은 이 검증과 별개다.
실험은 commit된 소스를 고정해 실행하며, 학습곡선과 mode coverage를 보고 판단한다.
외부 baseline 코드는 기존 구현을 보존했으며 v8 검증을 이유로 다시 학습하지 않았다.
