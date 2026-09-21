> Historical record. Final v2 is specified in [the final pseudocode](../../v2/PSEUDOCODE.md).

# Humanoid finite-mixture v2: 250K 조기종료 결과

2026-09-11 UTC. 새 finite-mixture 후보는 250K 비교에서 기존 continuous v2와
강한 historical OptiQ에 크게 뒤처져, seed 0–3 모두 조기종료했다.
이 후보로 안정적인 성능과 OptiQ 우위를 달성했다는 결론은 내리지 않는다.

## 같은 스텝의 성능 비교

230K, 235K, 240K, 245K, 250K의 평가 평균이다. 각 체크포인트는 stochastic
10 episodes이며, 네 학습 seed를 모두 포함했다. 마지막 평가 한 번의 점수가 아니다.

| 방법 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | 평균 ± 학습 seed SD |
|---|---:|---:|---:|---:|---:|
| 새 finite v2 | 1787.8 | 2683.1 | 1798.0 | 1193.7 | 1865.6 ± 613.8 |
| 완료된 continuous v2 | 4255.7 | 3693.9 | 2783.2 | 2300.0 | 3258.2 ± 881.0 |
| Matched OptiQ, 추가 uniform 0% | 1431.8 | 845.6 | 1252.3 | 978.0 | 1126.9 ± 264.6 |
| Historical OptiQ, 추가 uniform 10% | 3296.4 | 4213.9 | 2927.3 | 3328.1 | 3441.4 ± 546.2 |

새 후보는 continuous v2보다 42.7%, historical OptiQ보다 45.8% 낮다.
약한 matched control에 대한 우위만으로 목표 달성으로 처리하지 않았다.
Historical 비교에는 수집 정책과 코드 기록 차이가 있어 uniform 탐색의 인과적 효과를
분리한 ablation은 아니다. SD는 신뢰구간이 아니다.

![250K까지 네 방법의 평균과 seed별 결과](../../../outputs/v2_improvement/finite_report/step_0250000.png)

## 종료 판단과 결과 보존

200K 결과를 확인한 뒤 추가한 계산 예산 규칙에 따라, 250K에서 최근 다섯 평가의
4-seed 평균이 두 강한 reference 평균의 각각 70% 미만이면 모두 종료하도록 했다.
실제 낮은 쪽 reference 대비 비율은 0.5726이었다. 이 규칙은 200K 이후 정한
적응적 판단이며, 처음부터 등록한 기준이나 보지 않은 1M 성능에 대한 증명은 아니다.

네 seed의 250K actor/critic 체크포인트 총 8개를 확인한 후 supervisor가 순차 종료했다.
마지막 기록 스텝은 seed 순서대로 253376, 252210, 253113, 255370이다.
이 비동기 종료 차이는 250K 비교에 포함하지 않았다. 네 서비스는 STOPPED,
감시 서비스는 EXITED이며, 최종 평가 자동화는 `screen_incomplete`로 종료했다.
1M 완료나 최종 50-episode 평가 수행으로 표시하지 않았다. GPU compute process는
종료 후 조회에서 없었다. 학습 코드 및 설정의 SHA는 실행 당시와 동일하다.

실행 설정, 로컬 평가/학습 기록 및 체크포인트를 보존했다. 학습 재시작이나 push는
하지 않았다. 검증 기록: `outputs/v2_improvement/finite_review_0250000.json`.
W&B 프로젝트: <https://wandb.ai/OptiQ/optiq_mujoco_v2_finite_screen>.

## 종료 전 진단에서 보이는 문제

아래는 각 seed의 230K–250K 로그를 평균한 값이다. replay 상태에서의 진단이며,
전체 상태 공간에 대한 보장이나 성능 저하의 원인 증명은 아니다.

| 수치 | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|---:|
| ESS / 후보 64개 | 2.64 | 2.70 | 2.58 | 2.59 |
| density 항만 적용한 ESS 비율 | 4.59% | 4.55% | 4.71% | 4.81% |
| Q 항만 적용한 ESS 비율 | 14.84% | 18.48% | 14.44% | 14.77% |
| 같은 후보를 T=1로 재가중한 ESS 비율 | 4.71% | 4.65% | 4.80% | 4.88% |
| pre-tanh 분산 중 latent 평균 차이의 몫 | 0.93% | 0.80% | 0.90% | 0.93% |
| 조건부 Gaussian sigma 평균 | 0.20 | 0.20 | 0.20 | 0.20 |
| backup entropy 표본 평균 | −12.92 | −13.44 | −13.23 | −13.04 |
| 누적 guard 수락률의 구간 평균 | 48.89% | 48.68% | 49.28% | 48.80% |

온도를 높여 같은 후보를 재가중해도 ESS가 약 3/64에 머무른다. 이는 온도 변경만으로
현재 후보 집합의 집중을 해소하기 어렵다는 증거다. T=1로 실제 재학습한 실험은 아니다.
밀도 보정만으로도 ESS가 작고, Gaussian sigma는 상한 0.2에 닿아 있으며,
latent별 평균 차이가 제공하는 분산은 1% 미만이다. 후보 분포의 범위와 정책 표현의
제약을 다음 진단 대상으로 삼을 근거는 있지만, 어느 하나의 인과적 책임을 확정할 수는 없다.
연속 분포의 differential entropy가 음수인 것 자체는 오류가 아니다.

## 이론 보장과 현재 달성 수준

사용자가 정한 기준은 SAC와 같은 정확한 정책 평가를 가정한 이론적 보장과 실제
안정성·성능 검증이다. 정확한 Q만으로 임의의 W2/NLL projection이 개선되지는 않는다.
정확한 Boltzmann extraction과 분포 보존 distillation, 또는 모든 관련 상태에서
`E_new Q + T H_new >= E_old Q + T H_old`인 수락 조건이 이상적 개선 보장에 필요하다.
현재 replay 평균의 표본 guard는 그 상태별 조건을 증명하지 않는다.

Finite 혼합 정책의 전체 밀도를 계산해 finite-M 밀도 편향을 없앤 것은 사실이나,
그 변경으로 성능이 개선됐다는 실증 결과는 얻지 못했다. 이전에 완료한 continuous v2의
1M primary 평균은 5362.7로 matched OptiQ 5051.7보다 높고 historical OptiQ 5676.5보다
낮았다. 따라서 전체 개선 목표는 아직 미달성이다.
이론의 정확한 조건은 [보장 문서](../../v2/THEORY.md), 기존 완료 실험은
[확인 실험 결과](../../v2/RESULTS_KO.md)를 참조한다.
