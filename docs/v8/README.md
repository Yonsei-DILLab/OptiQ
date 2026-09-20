# v8 실행 안내

상태별 fresh raw-z 거리 OT + 조건부 SAC 구현이다. 전체 수식과 gradient 경로는
[ALGORITHM_KO.md](ALGORITHM_KO.md)에 정리했다. GMM40 runner와 RL이 같은 actor core를 호출한다.
현재 4개 GPU 실행 기록은 [GMM100K_KO.md](GMM100K_KO.md)에 정리했다.
통과한 테스트와 속도 측정 범위는 [VALIDATION_KO.md](VALIDATION_KO.md)에 기록했다.
이전 NLL 비용 검토 및 raw-z 거리 비용 선택의 근거는 [COST_REVIEW_KO.md](COST_REVIEW_KO.md)를 참고한다.

## 환경

Python 3.11, JAX/jaxlib 0.4.33, Flax 0.9.0, Optax 0.1.7, NumPy 1.26.4에서 검증한다.
기존 `requirements-mujoco.in` / `requirements-mujoco.lock`을 사용한다. PyTorch의
`2.4.1+cpu` wheel은 CPU index가 필요하다. 새 환경 설치 예:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mujoco.lock --extra-index-url https://download.pytorch.org/whl/cpu
```

GPU 명령은 CUDA 초기화에 실패하면 중단하며 CPU로 자동 전환하지 않는다.
아래 명령은 저장소 루트에서 실행한다. 출력 디렉터리는 실행마다 새 경로를 사용한다.

## CPU 검증

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/test_v8_transport.py tests/test_v8_actor.py \
  tests/test_v8_gmm.py tests/test_v8_rl.py -q
python -m gmm40.validate_v8 --platform cpu --quick --out /tmp/v8-cpu-preflight
```

CPU quick preflight는 작은 표본수의 코드 검증이며 성능 실험이나 full-shape GPU 속도 검증이 아니다.
RL 테스트에는 Ant에서 6 environment steps로 replay→soft TD→actor→checkpoint 연결을 확인하는 검증이 포함된다.

## GMM40 GPU 검증 및 학습

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m gmm40.validate_v8 --platform cuda --seed 0 --out /tmp/v8-gpu-preflight

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m gmm40.train_v8 --platform cuda --seed 0 --steps 100000 \
  --preflight /tmp/v8-gpu-preflight/preflight.json --out /path/to/results/v8-seed0
```

다른 seed는 `--seed`와 출력 경로를 바꾼다. 거리 epsilon은 `--epsilon 0.1`로 명시할 수 있다. Preflight는 source hash와 solver/clip 설정이
같으면 seed 간 공유할 수 있지만 실제 실행 GPU에서 CUDA 검증을 먼저 하는 것이 좋다.
이 인스턴스처럼 장기 작업을 supervisor로 관리하는 환경에서는 위 foreground 명령을
서비스에 등록한다. 다른 환경에서는 해당 환경의 job scheduler를 사용한다.

기본값:

| 설정 | 값 |
|---|---:|
| GMM Q / alpha | log GT / 1 |
| Batch lanes / raw latent 적분점 H | 256 / 4096 |
| Teacher 후보 → 사전 재표집 / actor 쿼리 | 256 → 16 / 16 |
| MLP / actor Adam | 256×2 / 3e-4 |
| 초기 σ / log σ 범위 / proposal floor | .5 / [-5,1] / .05 |
| OT min/max iterations / 상대 오차 | 10 / 2000 / 1e-3 |
| OT 거리 epsilon | .1, alpha와 독립 |
| Gradient clipping | 기본 없음; `--actor-max-grad-norm 2` 명시 가능 |
| 평가 표본수 / 기준 seed | 10000 / 20260917 |
| 평가·전체 checkpoint | 0, 1K, 이후 5K 간격 및 최종 |

Actor·Adam·RNG·설정 signature를 함께 저장한다. 재개도 새 디렉터리에 기록한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m gmm40.train_v8 --seed 0 --steps 100000 \
  --preflight /tmp/v8-gpu-preflight/preflight.json \
  --resume-checkpoint /path/to/parent/checkpoints/step_0050000.bin \
  --out /path/to/results/v8-seed0-resumed
```

설정/학습 source가 바뀐 checkpoint의 재개는 거부한다. OT 미수렴이나 비유한 학습 입력은
업데이트를 거부하고 마지막 유효 actor·Adam·RNG와 실패 진단을 저장한다.
`status.json`, `runtime.json`, `training.jsonl`, `checkpoint_audits.json`으로 확인할 수 있다.
`evaluations/step_*/samples.png`는 GT·정책·mode mass를 함께 표시하며 GT 중심은 점 위에 표시한다.
평가의 mode count는 40개 GMM component 중심 근처의 coverage이며 실제 density peak 수와 다르다.

## RL 연결

```bash
OPTIQ_PYTHON=python bash scripts/run_v8.sh 0 --check benchmark=ant
CUDA_VISIBLE_DEVICES=0 OPTIQ_PYTHON=python bash scripts/run_v8.sh 0 benchmark=ant
```

기본 Hydra profile은 `mujoco_v8`, alias는 `v8/final`이다. W&B project는 `v8`이다.
RL campaign은 GMM 검증 이후 별도로 진행한다. 전체 mixture entropy는 기존 fresh
16-component self-inclusive density로 근사하며, 조건부 actor loss와 구분한다.

OT는 고정된 raw z 좌표와 teacher의 pre-tanh 좌표 사이 거리만 사용한다.
4096개 source에 대한 actor forward는 하지 않는다. Gaussian은 teacher 후보 256개와
학습에 선택된 latent 16개에서 평가한다. 상태마다 teacher와 OT를 독립적으로 다시 계산한다.
이전 Gaussian-NLL-cost v8 checkpoint는 objective signature가 달라 재개할 수 없다.
