# DACER OFF 실행 및 entropy/noise 분석

기존 entropy sweep의 실행·대기 작업을 중지했다. 완료된 v1 −1/−.5 두 결과와 중단 로그는 보존했다. 새 v1~v4 OptiQ seed0 네 개는 `dacer.enabled=false`, T=1, dense reward, NovelD OFF로 실행했다. 학습 소스는 `484f92e7d6d34c964d85b4493ff17c5a9ebcf32e`, 브랜치는 `direct-gmm-trg-antmaze`, W&B는 OptiQ/antmaze다. 양 서버의 canonical HEAD와 frozen source를 일치시켰다.

등록 검증은 `registration-verification.json`에 있다. 4개 모두 8 learner updates 사전검사 및 dense replay/checkpoint 검증을 통과했다. 동일 RNG에서 train action과 direct-policy action이 실제로 일치하며 DACER noise/update가 0이다. random latent와 conditional sigma는 유지한다. 기존 T1 대조군과 초기 actor/critic 해시, 모델·optimizer·reward·평가 설정이 일치하며 DACER enabled만 변경됐다. 예산은 v1/v2 3,008,256, v3 4,008,448, v4 5,008,384 interactions. 256환경, batch4096, 256수집당 8업데이트, 256×3, actor3e-4/critic5e-4를 유지했다.

## 결론

**현재 DACER 추정치 기준으로 잡음을 증가시키려면 차원당 +0.60~+0.65가 합리적인 시험 후보이며, 하나를 고르면 +0.65(8차원 총 +5.2)를 먼저 검토한다. 최적값으로 검증된 것은 아니다.** 현재 late Ĥ/d≈.42~.53보다 높고, 초기에 .6 이상으로 올라가는 구간도 고려한 값이다. 음수 −1~−.1은 모두 현재 추정치보다 낮아 같은 감소 방향이다. 기존 설명에서 +.5도 후보라고 했지만, 전체 로그를 같은 Adam으로 재계산하면 +.5만으로는 초기 noise .027보다 크게 성장하지 않는다.

더 중요한 제약은 **조절이 약 320k environment interactions마다 한 번**이라는 점이다. interval10000은 learner update 단위이고, 256개 수집당 8회 학습하므로 이 환산이 된다. 초기 한 번을 포함해 전 학습에서 10~16회에 불과하다. Adam lr=.03로 log-alpha를 갱신하므로 목표를 크게 올린다고 잡음이 비례해서 커지지 않는다. 목표를 충분히 초과시키는 일정 방향의 기울기라면 대략 갱신 한 번당 exp(.03)≈1.03배다. .027→.1에는 단순 계산으로 약44회, 현재 간격으로 약14M interactions가 필요하다. 이는 대략적인 속도 설명이며 Adam의 실제 변동 기울기에 대한 상한은 아니다.

## 실제 완료 대조군 로그

| 환경 | 마지막 Ĥ/d | 추가 noise σ: 초기 → 최종 | 정책 σ 파라미터 평균 | DACER 갱신 수 |
|---|---|---|---|---|
| v1 | 0.529 | .027 → 0.0200 | 0.365 | 10 |
| v2 | 0.509 | .027 → 0.0200 | 0.358 | 10 |
| v3 | 0.421 | .027 → 0.0183 | 0.357 | 13 |
| v4 | 0.452 | .027 → 0.0167 | 0.358 | 16 |

Ĥ는 GMM joint-entropy proxy를 action_dim=8로 나눈 값이다. 전체 학습에서는 첫 값 약.37에서 .62~.69까지 올랐다가 후반 .42~.53으로 내려왔다. 표의 Ĥ/noise는 마지막 DACER 갱신값이며, 정책 σ는 마지막100k 로그의 평균이다. σ는 truncated Gaussian의 파라미터이므로 실제 bounded action 표준편차와 다르다.

## 목표만 바꾼 계산

기존 Ĥ 관측열을 고정하고 동일한 scalar Adam(초기alpha.27, lr.03, b1=.9, b2=.999, eps1e-8)을 재생했다. 원래 −.9를 넣으면 실제 noise 로그를 2e-7 이내로 재현한다. 아래는 최종 추가 noise 표준편차다.

| 환경 | h*=+.50 | h*=+.60 | h*=+.65 |
|---|---|---|---|
| v1 | 0.0235 | 0.0278 | 0.0307 |
| v2 | 0.0264 | 0.0333 | 0.0346 |
| v3 | 0.0241 | 0.0341 | 0.0368 |
| v4 | 0.0236 | 0.0348 | 0.0389 |

초기 noise는 .027이다. **정책/방문상태가 바뀌지 않는 가상 계산이며, 새 목표로 실제 학습한 결과나 예측구간이 아니다.** +.65도 .031~.039 정도여서 증가 폭은 작다. 현재 요청대로 실행한 실험은 DACER OFF 네 개뿐이며 양수 목표 실험은 실행하지 않았다.

## baseline과 action-level 지표

저장된 최종 체크포인트 SHA256을 확인하고 CPU에서만 forward sampling했다. 각 방법 자신의 v3 replay에서 24개 상태를 뽑아 상태당 400개 행동을 샘플링했다. 아래 표준편차는 상태 내 분산을 좌표·상태에 평균한 뒤 제곱근을 취한 값이다. 서로 다른 replay 상태이므로 원인·성능의 공정한 비교로 단정하지 않는다.

| 방법 / v3 | 상태 내 action 표준편차 | 추가 정보 |
|---|---|---|
| OptiQ | 0.433 | random z + conditional sigma; DACER 추가 noise 제외 |
| SAC | 0.151 | tanh Gaussian 실제 log-density MC H/d=-1.134; 목표 −1 |
| DIPO | 0.183 | native diffusion 출력; reverse noise 포함 |
| DIPO + 학습 탐색 | 0.338 | 환경별 mixed Gaussian σ=.05~.6 (원시 RMS .362), clip 적용 |
| MFPO | 0.140 | 학습된 density의 H/d=-0.507; 목표 −.5 |

이 진단에서는 **OptiQ의 직접 정책 행동 분산이 이미 작지 않다.** 낮은 추가 DACER noise만 보고 전체 행동 탐색이 좁다고 말할 수 없다. SAC alpha는 noise σ가 아니라 entropy reward 가중치이며, SAC의 −1이나 MFPO의 −.5를 현재 GMM proxy의 적정 목표로 그대로 옮길 수 없다. MFPO의 density 추정치와 GMM proxy도 서로 다른 양이다.

같은 v3 OptiQ replay 상태로도 네 정책을 샘플링했으며 그 결과는 analysis.json에 보존했다. SAC의 pre-tanh std가 커져 극단 포화되는 OOD 상태가 있어, 이 공통 상태 비교만으로 baseline의 평소 entropy를 판정하지 않았다. 위 표에는 별도로 측정한 각자 replay 결과를 사용했다.

네 OptiQ 모델에서 추가 noise std=.02는 action spread를 약0.1%만 변화시킨다. .1에서는 약2%, .2에서는 약6~7%, .35에서는 약15~18% 늘어난다. 이 역시 frozen-policy forward 진단이며 새 궤적이나 성공률 개선 결과가 아니다. .35에서는 clip으로 ±1 경계에 놓이는 좌표가 약13~15%이므로, 크게 넣은 noise가 그대로 유효한 탐색이 된다고 볼 수 없다.

## entropy 추정의 범위와 권고

현재 DACER는 3개 full-covariance Gaussian과 상태당200samples로 `H(component)+Σw H(Gaussian)`을 계산한다. 이는 **맞춘 GMM의 marginal entropy에 대한 상계**이지 실제 bounded policy entropy의 정확한 값이 아니다. 별도 샘플링에서 joint−marginal 차이는 OptiQ 약.018~.021/dim이었다. 이론상 최대 ln3/8≈.137/dim와 별도로 fitting 오차가 존재한다. fitted GMM에서 나온 행동 중 약37~39%는 8좌표 중 적어도 하나가 [-1,1] 밖에 있었다. 실제 policy 샘플은 모두 범위 안이다. 따라서 +.65는 **이 추정기 스케일에서의 제어 후보**이며, 실제 entropy를 uniform의 ln2에 맞춘다는 뜻이 아니다.

다음 판단은 학습 중 실제 extra-noise std, conditional sigma, clip fraction, critic 상태와 함께 공간 coverage·분기점 양쪽 방문·같은 시작상태에서의 성공 경로 비율을 같이 보아야 한다. 큰 action entropy가 시간적으로 일관된 두 경로를 보장하지 않는다. 현재 OptiQ는 plain TD를 유지하고 DACER는 behavior-only 조절이므로 SAC/MFPO의 MaxEnt 목적함수와도 다르다. 완료된 v3 모델의 높은 action spread와 경로 편중이 함께 나타난 점은 단순한 '행동 잡음 부족'으로 설명하기 어렵다.

권고는 (1) 현재 OFF 대조군과 기존 ON을 비교해 실제 영향 확인, (2) 이후 DACER를 다시 쓸 경우 +.60/.65 후보와 **갱신 간격**을 별도 요인으로 평가, (3) 목표값만 크게 올려 해결하려 하지 않는 것이다. 갱신 간격·noise scale·sigma 상한은 이번 실행에서 바꾸지 않았다.

근거: 보존 로그27개, SHA256 검증한 완료 체크포인트7개, 각 모델 24상태의 CPU forward sampling. 모든 결과는 training seed0 하나이며 상태 표본이 작다. 코드 기준은 [DACER 원문 Eq.15/16](https://proceedings.neurips.cc/paper_files/paper/2024/file/6174c67b136621f3f2e4a6b1d3286f6b-Paper-Conference.pdf), [DDiffPG 공식 SAC](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/sac.py), 현재 보존 source의 regulator.py 및 native baseline config다. 논문 공식 환경별 최적 entropy를 찾았다는 의미는 아니다.

![로그와 baseline 분석](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_t1/entropy_noise_audit.png)

![고정 정책 noise 진단](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_dacer_off_t1/frozen_policy_noise_grid.png)
