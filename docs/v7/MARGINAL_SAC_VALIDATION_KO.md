# 중단한 v7 marginal SAC 비교 실험 기록

> 2026-09-19 사용자 요청으로 이전 조건부 entropy＋OT 배정 loss를 복원했다. 이 문서는 비교 실험의 당시 구현·검증 기록이며 현재 기본 설정을 설명하지 않는다. 현재 상태는 [CONDITIONAL_RESTORE_KO.md](CONDITIONAL_RESTORE_KO.md)와 [ALGORITHM_KO.md](ALGORITHM_KO.md)를 따른다.

2026-09-19 비교 실험에서 `ot_marginal_sac`은 기존 `ot_conditional_sac`의 조건부 entropy와 OT 배정항을 전체 H=4096 고정 normal latent bank의 **actor mixture density**로 바꾸었다.

\[
\hat\pi_\theta(a\mid s)=\frac1H\sum_i\pi_\theta(a\mid s,z_i),\qquad
\hat L=\frac1{16}\sum_k\operatorname{sg}\!\left[\frac{1/H}{\sum_jP_{i_kj}}\right]
\big[T\log\hat\pi_\theta(a_k\mid s)-Q(s,a_k)\big].
\]

별도의 `-T log Pr(i|a,s)` 항이 없으며, entropy 밀도 계산에는 OT로 선택된 16개만이 아닌 전체 4096개 current actor Gaussian이 들어간다. Teacher의 sigma 하한을 actor 밀도에 적용하지 않는다. Tanh Jacobian과 GMM의 물리 좌표 배율40 Jacobian을 포함한다. 전체 밀도와 생성 행동에 autodiff를 적용한다.

Teacher256개, importance 재표집256→16, OT4096×16, actor16쌍, batch256, actor LR3e-4, dual LR1e-4 및 매 actor당 dual Adam1회는 그대로다. GMM은 alpha=T=1, epsilon=.1이다. 공유 상태와 공유 bank의 actor Gaussian4096개를 한 번만 평가하고 batch lane에 재사용하며, 일반 경로와 loss·gradient가 일치한다.

RL soft TD도 같은 고정 H bank에서 행동을 표집하고 같은 mixture 밀도를 합산한다. 기존 다른 profile의 TD/IDAC는 보존했다. 수집·평가는 기존 continuous normal prior를 유지하므로, 학습의 finite bank와 continuous policy의 적분 근사 차이는 남는다.

## 구현 검증

- Actor/dual CPU 검사39개 통과: 독립 Gaussian 합산으로 loss·gradient 검증, 표집되지 않은 component의 밀도 gradient, source importance 기대값 항등식, H=1, sigma floor 미적용, 공유 상태 경로 일치, teacher/RNG/dual Adam 상태.
- Config/RL CPU 검사63개 통과: full-bank soft TD 수치 oracle, target stop-gradient, legacy profile 유지, canonical route 검사.
- Ant 동작 검증: H4096, batch2, 환경8스텝에서 actor·critic·dual 각각6회 업데이트. 두 mu-only 평가와 체크포인트 저장·재개 검사. 성능 실험이 아니다.
- GMM CPU 통합검사: shared core와 adapter의 actor·dual·RNG 차이0, checkpoint1→2 재개 차이0, 다른 latent bank와 과거 actor objective checkpoint 거부.

[GMM CPU preflight 원자료](/root/optiq-experiments/v7/gmm40-validation/marginal-sac-fullbank-cpu-preflight.json)

각 테스트 파일의 최초 실행에서 새 metric 의미에 맞지 않는 기존 assertion을 수정하고 해당 검사를 재실행했다. 위 개수는 수정 후 통과한 서로 다른 검사 수이며 하나의 통합 실행 로그 개수를 뜻하지 않는다.

## GPU 검증과 새 100K 실행

[GPU 검증 폴더](/root/optiq-experiments/v7/gmm40-validation/marginal-sac-fullbank-gpu-20260919T231856Z): seed0, batch256, H4096, teacher256→16, actor16쌍으로200회 업데이트 완료. 모든 training 지표가 유한하며 actor·dual·각 Adam count와 checkpoint가 일치했다.

Warm update는 **3.27ms**였다. 같은 GPU 계열에서 과거 조건부 loss의 seed0 100K 중앙값은 **1.79ms**, 과거 OT-NLL H4096·256→16 seed0 기록은 약 **2.04ms**다. 약1.83배/1.60배의 시간이다. 새 smoke는 compile 제외 warm100회 구간 한 개로, 장기 실행 속도는 각 run의 `runtime.json`으로 다시 확인해야 한다. Evaluation·checkpoint 시간은 제외한다.

[새 campaign manifest](/root/optiq-experiments/v7/gmm40-training/marginalSAC-P256-H4096-K16-T1-e01-20260919T232101Z/jobs.json)

| GPU | Seed | 목표 actor/dual 업데이트 | epsilon | alpha |
|---:|---:|---:|---:|---:|
| 0 | 0 | 각100K | .1 | 1 |
| 1 | 1 | 각100K | .1 | 1 |
| 2 | 2 | 각100K | .1 | 1 |
| 3 | 3 | 각100K | .1 | 1 |

각 실행은 처음부터 시작했다. 0, 1K와 이후5K마다 평가·전체 체크포인트를 저장했고, 같은 seed의 조건부 loss와 동일한 평가 RNG·표본10,000개·target을 사용했다. 사용자 요청에 따른 원래 목적함수 복원 시 seed0은83K, seed1–3은85K에서 SIGTERM으로 중단했다. 네 실행 모두 해당 step의 actor·dual·각 Adam·RNG 전체 체크포인트와 SHA-256을 검증했다. 100K 완료 결과가 아니다. Supervisor worker들은 STOPPED다.

## 결과 해석

목적함수는 finite-bank marginal SAC와 일치한다. 이는 신경망의 전역 최적화나 GMM40 모든 모드의 복원을 보장하지 않는다. OT와 source 보정이 정확할 때 OT는 기대 목적함수를 바꾸지 않고 표집 분포·분산에 영향을 준다. 발견한 teacher 위치를 직접 NLL 목표로 삼지는 않는다. 작은 source 질량에 따른 보정 분산도 유지되므로 지표를 계속 기록한다.

과거 조건부 loss의 checkpoint·소스·평가는 원래 결과 폴더에 보존되어 있다. 새 checkpoint signature에 목적함수와 density 방식을 기록해 잘못된 재개를 막는다. 외부 baseline 소스와 과거 결과는 수정하지 않았다.
