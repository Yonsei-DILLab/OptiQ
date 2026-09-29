**DACER 구현 확인 및 pull 결과 — 2026-09-21**

서버180의 `/home/heechan/OptiQ-direct-gmm-trg`, branch `direct-gmm-trg`에 원격 최신 변경을 pull했다. 로컬 `84f1e0a`와 원격 `4ca6947`이 갈라져 있어 fast-forward 대신 기존 이력을 보존하는 merge로 완료했다. 최종 HEAD는 `3838cf710dcf36bde1ff9aaffe30ccfdd7d57858`이며 작업 트리는 깨끗하다. 실행 중인 실험 소스 `/home/heechan/OptiQ-ops/sources/84f1e0a884349d6c4b0dae521839a8d4e5f46437`는 원래 커밋을 유지한다.

DACER 구현은 `analysis_tools/experiments/20260921_gmm_trg_sweep/regulator.py`의 `BehaviorRegulatedOptiQ`다. 같은 폴더의 `train.py`가 기존 runner의 OptiQDIME 클래스를 이 subclass로 연결한다. 이 폴더의 소스는 merge 후에도 원격4ca6947과 동일하다. 기존 `20260921_temp_beta/train_sweep.py` 진입점은 이 subclass를 사용하지 않는다.

**수집 행동**

기존 actor에서 z와 조건부 Gaussian 잡음을 포함한 bounded policy action을 뽑고, 정규화 action 공간에서 `clip(a_policy + lambda * alpha * epsilon, -1, 1)`을 실행한다. 실제 실행한 noisy normalized action을 replay에 저장한다. Warm-up 이전에는 기존 random action을 그대로 쓴다. Alpha는 run당 하나의 양수 스칼라이고 log alpha를 최적화한다. Actor의 state/z별 log sigma와 별도다.

**Entropy 추정 및 갱신**

Replay state256개 각각에서 현재 policy action200개를 뽑고 추가노이즈 및 clipping까지 적용한다. 따라서 entropy 측정 한 번에51,200개의 action을 생성하고 state별로256번 GMM을 fitting한다. GMM은 sklearn GaussianMixture, 성분3개, full covariance, random_state42다. 성분 가중치를 w, 각 Gaussian entropy를 H_k라 할 때 proxy는 `-sum(w log w) + sum(w H_k)`이고 이를 state에 걸쳐 평균낸다. 이것은 성분 label과 action의 joint entropy이며 mixture marginal의 정확한 entropy가 아니다. Clipping 경계의 확률질량도 별도로 모델링하지 않는다.

목표는 `H_target=-0.9*action_dim`, Ant=-7.2, Humanoid=-15.3이다. `H_proxy-H_target`을 log alpha의 gradient로 Adam에 전달한다. 목표보다 entropy가 낮으면 추가 탐색을 키우는 방향이다. 이 과정에서 actor에 entropy gradient를 전달하지 않는다.

| 항목 | 기본값 |
|---|---|
| enabled | false |
| behavior_only | true; 이 진입점은 true를 요구 |
| initial_alpha | .27 |
| alpha_lr | .03, Adam |
| interval_updates | 10,000 learner updates |
| 최초 갱신 | 첫 training call의 update count0; 기본 warm-up5k 이후 |
| components / samples | 3 / state당200 |
| entropy_seed | 42 |
| noise_scale 공통 기본 | .1 |
| campaign noise_scale override | Humanoid/HalfCheetah .15, 나머지 .1 |
| 초기 추가 noise std | Ant .027, Humanoid .0405 (campaign override 사용 시) |

**연결 범위와 기록**

추가 noise는 환경 수집에만 적용된다. Teacher/proposal/density correction/Direct GMM NLL, TD next action 및 dual mu-only 평가에는 추가하지 않는다. 기존 학습 batch RNG를 보존하도록 entropy용 replay sample 전후에 NumPy RNG를 복구하고, policy 진단 샘플에는 별도 JAX key를 쓴다. 수집 noisy action을 통해 replay 데이터는 달라진다.

W&B/logger의 `exploration/entropy_proxy`, `target_entropy`, `alpha`, `noise_std`, `updates`, `gmm_converged_fraction`, `estimation_seconds`, `clip_fraction`과 `dacer_regulator.json`에 진단값을 남긴다.

확인한 구현 한계: alpha의 명시적 상하한은 없고 finite 여부만 검사한다. GMM이 일부 미수렴해도 수렴 비율을 기록하고 유한한 추정값으로 갱신을 진행한다. 추가 noise만 조절하므로 actor 자체의 sigma/entropy를 목표 이하로 낮추는 제어는 아니다. 일반 actor/critic checkpoint 저장에는 regulator의 log-alpha, Adam state, RNG가 포함되지 않고 JSON은 진단값이므로, 완전한 regulator 재개 상태 저장·복구는 구현되어 있지 않다.

**검증**

서버 기본 venv에는 sklearn이 없어 최초 실행은3 passed/2 failed(ModuleNotFoundError)였다. 실행 중 학습환경을 수정하지 않고 `/tmp/optiq-dacer-audit-deps`에 scikit-learn1.5.2, joblib1.4.2, threadpoolctl3.5.0을 격리 설치하고 PYTHONPATH로 지정했다. CPU에서 기존 `test_sweep.py`를 재실행하여 **5 passed,14.49초**를 확인했다. 검증 범위는 설정 유지, entropy/alpha 방향 및 clipping, 축소된 단기 학습의 regulator 실행, 마지막100k 계산, 등록된 실험 계획의 개수/중복 검사다. 장기 성능이나 논문의 원본 구현과 완전한 동등성을 검증한 것은 아니다. 새 training campaign은 시작하지 않았다.
