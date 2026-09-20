# v5 Direct GMM 검증 기록 — 2026-09-20

구현 커밋 `2532303a28e0203361c17a9f5b280e5f302d0c45`, 검증 정리 커밋
`3b03801d15cd330d3b0b86b6f067b890826db286`을 실제 RTX 5090에서 검증했다.
이 기록을 추가한 후속 커밋은 README/설명/docstring만 변경한다.

| 검증 | vast-heechan-199 | vast-heechan-180 |
|---|---|---|
| Direct GMM + 기존 v5/proposal/distillation/exploration 검사 | 45개 고유 검사 통과 | 45개 고유 검사 통과 |
| 기본 mujoco 환경의 Direct GMM 전체 검사 | 13 passed, 74.51초 | 13 passed, 36.84초 |
| 실제 Ant: 8 환경 step, 6 actor/critic update, 두 평가 모드 | 통과 | 통과 |
| GMM40 Direct GMM / OT OptiQ / SAC / DIPO / MEow / MFPO | 6개 모두 통과 | 6개 모두 통과 |
| 각 adapter: GPU update, checkpoint 다음 update 재현, 유한 표본 | 통과 | 통과 |
| GMM40 fixed-Q CLI 및 optimizer counter audit: 2 update | 통과 | 통과 |
| GMM40 navigation CLI 및 actor/critic counter audit: 2 update | 통과 | 통과 |
| Direct GMM 2-GPU pmap: 동기 replica, checkpoint 재개 | 통과 | 통과 |

45개 검사는 `test_direct_gmm.py`, `test_v5.py`, `test_conditional_proposal.py`,
`test_distributional_distillation.py`, `test_v5_exploration.py`로 구성된다.
최초 실행은 44 passed와 테스트 종료 처리 오류 1건이었다. Ant의 학습·평가
assertion은 이미 통과했지만 monkeypatch된 함수에서 `clear_cache()`를 호출했다.
원래 JIT 함수를 보존하여 정리하도록 수정한 뒤 해당 검사와 Direct GMM 전체
13개를 두 서버에서 재실행해 통과했다. 실패 로그도 남겼다.

## 수식·코드 확인

- `direct_gmm_nll` 함수 AST가 heejoon의 원본과 동일하다.
- SciPy Gaussian mixture log likelihood와 독립 analytic responsibility gradient가
  일치한다. 평균과 log-sigma head 모두 검증했다.
- teacher u/w 및 critic 쪽 역전파는 0이고, 학생 쪽 logsumexp는 미분된다.
- action Jacobian을 포함/생략한 학생 gradient가 같다.
- 실제 proposal의 floor, joint mixture, tanh Jacobian, sampling을 독립 수식과 비교했다.
- v5와 teacher/RNG/density/ESS가 같고, Sinkhorn을 호출하면 실패하도록 바꿔도
  Direct GMM update 및 실제 Ant 경로가 통과한다. OT epsilon/iteration을 바꿔도
  Direct GMM 결과는 같다.
- 기존 policy/proposal/critic 및 common/diffusion 코드는 보존했다.
- GMM40 target와 SAC/DIPO/MEow/MFPO 구현, 외부 baseline pin은 v5-gmm40 원본과 동일하다.

## 환경 및 병렬 실행 범위

기본 환경은 `/home/heechan/.venv-optiq-mujoco`다. 기존 v5와 동일한
JAX/JAXlib 0.4.33, Flax 0.9.0, Optax 0.1.7, NumPy 1.26.4를 사용한다.
PyTorch는 2.4.1+cpu이며, JAX GPU 계산에는 문제가 없다.
이 환경의 NCCL은 2.31.2이고, 두 서버에서 2-GPU pmap을 검증했다.
서버 간 8-GPU 분산 학습을 검증한 것은 아니다.

CUDA PyTorch baseline은 별도 `/home/heechan/.venv-optiq-gmm40`의
PyTorch 2.7.1+cu128, NCCL 2.26.2를 사용한다. 단일 GPU에서 모든 adapter가
정상 작동하지만, 해당 환경의 JAX pmap은 두 서버에서 NCCL 초기화 timeout이
재현됐다. `NCCL_CUMEM_HOST_ENABLE=0`만으로 해결되지 않았다.
패키지를 임의로 업그레이드하지 않고, 검증된 기존 v5 환경으로 JAX 병렬 실행을
분리했다. 여러 CUDA 라이브러리와 Torch build가 함께 다르므로 NCCL 버전 하나만을
근본 원인으로 확정하지 않는다.

```bash
# MuJoCo / Direct GMM / JAX pmap: 기본
source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm

# CUDA PyTorch baseline: 단일 GPU
source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm gmm40
# GPU lock wrapper로 선택할 때
OPTIQ_RUNTIME_OVERRIDE=gmm40 /home/heechan/OptiQ-ops/run-gpu.sh 0 \
  --branch v5-direct-gmm python -m gmm40.run --method sac --name sac_s0
```

Docker 호환을 위해 activation은 `NCCL_CUMEM_HOST_ENABLE=0`을 기본 설정한다.
이는 [NVIDIA 공식 설명](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html)의
cuMem host allocation 대체 경로이며, 구 환경 timeout의 충분한 해결책은 아니었다.

## 결과 보관

두 서버의 `/home/heechan/OptiQ-ops/direct-gmm-report.json`에 최종 branch SHA,
각 환경, 테스트 결과, baseline SHA, checkpoint audit를 기록한다.
상세 로그는 `/home/heechan/OptiQ-ops/validation/v5-direct-gmm/` 및
`direct-gmm-pytest*.log`에 있다. 최종 배포 이후 code checkout은 모두 clean이다.
Mac에도 `optiq-vast-setup/reports/<SSH 별칭>/`으로 보고서와 로그를 복사한다.

모든 실행은 짧은 **validation**이다. 장기 학습 결과, 수렴·성능 우위·mode coverage를
주장하지 않는다. 기존 실험, 출력 디렉터리, v5/heejoon/v5-gmm40 브랜치는 보존했다.
