# 조건부 Boltzmann 추출 방식 복원

2026-09-19 사용자 요청: 목표는 SAC와 동일한 업데이트의 구현이 아니라 Boltzmann policy를 제대로 모사하는 알고리즘이다. 따라서 전체 mixture SAC 비교 변경을 되돌리고 이전의 조건부 entropy＋OT 배정 loss를 현재 기본값으로 복원했다.

## 현재 알고리즘

\[
\widehat L=\frac1K\sum_{k=1}^{K}
\operatorname{sg}\!\left[\frac{1/H}{\sum_jP_{i_kj}}\right]
\left[T\log\pi_\theta(a_k\mid s,z_{i_k})-Q(s,a_k)
-T\log\Pr(i_k\mid a_k,s)\right].
\]

- 기본 objective ID: `ot_conditional_sac`.
- H4096 고정 normal latent 적분점, fresh teacher256개, 후보마다 독립 latent에서Gaussian 행동 하나, full proposal density로 W 보정한 뒤256→16 재표집.
- OT4096×16에서 선택된16개 latent의 actor 출력만 학습한다. 전체4096개 actor mixture density는 계산하지 않는다.
- Teacher용 actor 평가256개와 학습용 actor 평가16개를 유지한다. Teacher sigma 하한 .05, actor sigma 범위·초기값·MLP는 이전 설정이다.
- Q parameter와 OT potential은 고정하되 새 행동을 통한 Q·배정함수 gradient는 모두 통과한다.
- Source importance는 clipping·self-normalization 없이 유지한다.
- Persistent dual은 기존256×2 ReLU MLP와 Adam1e-4, actor당1회 업데이트를 유지한다.
- Actor LR3e-4, LR/UTD/FREQ 변경 없음. 기본 epsilon=.1, GMM alpha=T=1.
- RL soft TD는 이전 random16-component self-inclusive mixture 밀도 추정으로 복원했다. 수집·평가도 이전 설정이다.

이 알고리즘은 OT로 나눈 조건부 Boltzmann 목표를 학습한다. Population source 질량이 prior와 맞고 각 조건부를 충분히 잘 표현·적합할 때 전체 Boltzmann 분포를 복원할 수 있다. Marginal SAC 목적함수와의 동일성이나 신경망 SGD의 전역 수렴을 주장하지 않는다.

## 소스 복원과 검증

복원 원본은 변경 전에 저장된 [조건부 실행 소스](/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-eps1-20260919T225957Z/seed0/source)다. 이 원본은 epsilon을 CLI로 받으며 기본값은 .1이다.

`conditional_sac.py`, `semi_implicit.py`, `persistent_transport.py`, `algorithm.py`와 GMM adapter·runner·validation을 원본 byte 그대로 복원했다. 원래 CPU preflight가 검증한15개 소스 SHA-256이 모두 일치한다. CLI·config·metadata·검증 테스트도 조건부 방식으로 되돌렸다.

Actor/dual CPU 검사32개와 RL/config 검사62개, 총94개가 통과했다. 선택된16개 actor forward, 조건부 density, 행동을 통한 배정 gradient, source 보정의 기대값, persistent dual·Adam 상태를 확인했다. RL 검사에는 H4096·IDAC16을 사용하는 Ant8스텝 동작 검증과 checkpoint·평가 검증이 포함된다. Launcher 검사도 통과했다.

이전 조건부 seed0의1K checkpoint를 불러와 actor·dual 업데이트가 각각1000임을 확인했다. Marginal SAC83K checkpoint는 설정 signature 불일치로 거부됐다. 원래 checkpoint 파일을 변경하거나 학습 업데이트를 수행하지 않았다. [복원 감사 원자료](/root/optiq-experiments/v7/conditional-restore-20260919T232802Z/audit.json)에 해시·검증 결과를 저장했다.

## 비교 변경과 결과 보존

변경 전의 marginal SAC 코드·설정·문서·테스트 전체는 [복원 전 아카이브](/root/optiq-archives/v7-before-conditional-restore-20260919T232802Z)에 보존했다. 외부 baseline 코드는 수정하지 않았다.

[Marginal SAC campaign](/root/optiq-experiments/v7/gmm40-training/marginalSAC-P256-H4096-K16-T1-e01-20260919T232101Z/jobs.json)은 사용자 요청에 맞춰 중단했다.

| Seed | 중단 업데이트 | 최종 체크포인트 |
|---:|---:|---|
| 0 | 83,000 | actor·dual·Adam·RNG 전체 저장, 해시 검증 |
| 1 | 85,000 | 동일 |
| 2 | 85,000 | 동일 |
| 3 | 85,000 | 동일 |

기존 조건부100K 결과와 checkpoint는 그대로 유지한다. 이번 요청으로 새 장기 실험을 시작하거나 기존 조건부 실험을 재개하지 않았다.
