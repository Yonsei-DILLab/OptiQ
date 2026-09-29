# AntMaze 경로 유지: 알고리즘 변경 없는 설정·환경 대안 감사

## 결론의 범위

**가능성은 있지만, 지금까지 검증된 단일 설정으로 공식 v3/v4의 동일 full start에서 두 성공 경로를 장기 유지했다고 말할 수 없다.** 공식 v1에는 예외가 있다. Dense `−d(next)`, T=3, DACER/NovelD OFF, 256×3, 256env/batch4096/8 updates per vector step, γ=.99, 3M 예산으로 학습한 seed0 정책은 **같은 원점 full state**에서 direct policy 100회 중 위20·아래71회 성공(실패9), 새 평가 난수로도 위26·아래66회 성공(실패8)이었다. 여러 training seed 재현은 아직 없다. [v1 동일 상태 보고](../../antmaze_v1_T3_origin_supplement/report/REPORT_KO.md).

v3/v4에 관한 아래 결과는 서로 다른 reward·T·DACER·γ 조건의 독립 캠페인이다. 서로 다른 조건의 수치를 한 factorial로 간주하지 않는다. 통로 **진입**과 목표 **성공**은 구분한다.

| 변경 축 | 저장 결과 | 판정 |
|---|---|---|
| 온도 | Geodesic v4 T=3은 250k direct100회 위38/아래45/미진입17이지만 성공0; T=10은 미진입98/100. γ=.999인 v3 T=3은 250k 왼쪽20/오른쪽80에서 550k 오른쪽39/40으로 집중. | 적당한 T 상승은 초반 양쪽 진입을 만들 수 있지만 유지나 양쪽 성공은 미입증. 너무 큰 T는 이동 자체를 해칠 수 있다. [v4 온도](../../antmaze_v4_geodesic_temperature_250k/report/REPORT_KO.md), [v3 장기](../../antmaze_gamma999_T3_retention_1m/report/REPORT_KO.md). |
| 할인율·시간 제한 | v3 γ=.999 단독은 250k 양쪽 진입 후 1M 왼쪽99/100. γ=.99999·T3의 장기 실험도 550k 오른쪽40/40. 같은 정책의 episode limit700→1400은 v3 성공14→66/100으로 늘었지만 통로 진입은 왼쪽18·오른쪽80→왼쪽18·오른쪽81이었다. | 더 긴 가치 horizon이나 평가 시간은 일부 성공을 돕지만 경로 선택의 편중을 풀지 못했다. [γ=.999](../../antmaze_gamma999_retention_1m/report/REPORT_KO.md), [γ=.99999](../../antmaze_normalized_discount_retention_1m/report/REPORT_KO.md), [시간 제한](../../antmaze_paired_horizon_diagnostic/report/REPORT_KO.md). |
| σ 범위·teacher 탐색폭 | log σ 상한 −1/0/1/2/3의 250k 비교에서 양쪽 **성공**은 확인되지 않았다. v3 teacher floor를 exp(−5)→1로 늘리자 ESS는 약19→34/64가 됐지만 650k 경로는 오른쪽40/40이다. | 후보 유효 수 증가와 경로 유지가 동일하지 않다. 상한을 넓히는 단독 처방은 뒷받침되지 않는다. [σ 상한](../../antmaze_actor_sigma_higher_v34_250k/report/REPORT_KO.md), [teacher floor 장기](../../antmaze_teacher_floor_retention_1m/report/REPORT_KO.md). |
| 병렬 env 수·fixed64 latent | 256env→32env에서 같은 transition/update 예산의 v4 250k가 양쪽71/29에서 아래100/100으로 바뀌었다. Fixed64 codebook은 매 행동 다시 선택하며 v3 250k 양쪽 진입 소수를 유지했지만 양쪽 성공은 아니다. | 업데이트 시점과 선택 방향은 바뀔 수 있으나, 확인된 해결책은 아니다. [env 수](../../antmaze_env32_v34_250k/report/REPORT_KO.md), [fixed64](../../antmaze_fixed64_v34_250k/report/REPORT_KO.md). |
| Critic LR·target τ·actor 지연·보상/T 동시 0.2배 | 같은 `100(d_t−d_{t+1})` v3/v4 500k 12-run 감사에서 어느 조건도 두 경로를 안정적으로 유지하지 않았다. | 이 값들 하나의 조정으로 현재 소실이 해결됐다는 증거는 없다. [critic dynamics](REPORT_KO.md). |
| Replay 용량·UTD | `100(d_t−d_{t+1})` 이전 control은 500k 총508,416개 전이만 수집해 1M buffer의 삭제·덮어쓰기가 **0**인데도 경로가 한쪽으로 갔다. 별도 UTD=1 현재 checkpoint의 시작 상태 64 후보 진단에서는 v4 100k의 두 후속 난수 반복 모두 위쪽64/64였다. | 용량 증설은 초기 붕괴를 막을 수 없다. UTD 증설도 그 자체로 두 경로를 보장하지 않는다. 최신 UTD=1 진단은 다른 256×2 학습 프로필과 초기 checkpoint이므로 500k control과 직접 우열 비교는 하지 않는다. [전체 replay 감사](../../antmaze_replay_hypothesis_20260925/REPORT_KO.md), [UTD=1 후보 진단](../../antmaze_utd256_euclidean_20260925/diagnostics/teacher_probe_results.json). |
| 보상·시작 상태 | Euclidean `100Δd`에서 원점의 두 목표 거리: v3 둘 다16.97056m, v4 둘 다16.49242m. 그럼에도 한쪽에 집중됐다. Geodesic/Euclidean, step penalty/bonus 유무를 바꾼 기존 실험에서도 v3/v4 장기 두 성공 경로는 확인되지 않았다. 공식 v1의 랜덤 시작 두 성공 경로는 동일 상태 증거가 아니므로 별도 원점 재평가가 필요하며, 위 T3 실험에서는 실제로 확인됐다. | 출발점 목표거리 비대칭이나 success bonus가 유일한 원인은 아니다. [고정 시작 기록](../../antmaze_progress100_fixed_history/REPORT_KO.md), [보상 factorial](../../antmaze_progress100_factorial/metric_reports/20260924T140516Z/REPORT_KO.md), [v1 원점](../../antmaze_v1_T3_origin_supplement/report/REPORT_KO.md). |

## 권장되는 다음 검증 — 새 실험은 등록하지 않음

1. **재현 가능한 양쪽 성공의 최소 실증:** 위 공식 v1 T3+dense 설정을 변경 없이 서로 독립적인 training seed 여러 개에서 다시 학습하고, 매 seed의 같은 **원점 full state**에서 random-z+conditional-σ 100회 이상으로 두 경로의 성공률과 후반 유지 여부를 분리해 평가한다. 기존 한 seed는 가능성의 증거이지 일반적인 보장이 아니다.
2. **현재 progress reward에서 순수 온도 검증:** `100Δd`, no bonus/step, DACER/NovelD OFF, 고정 시작 v3/v4에 T만 1→3으로 바꾸고 충분히 후반까지 추적한다. 기존 T3 성공/실패는 주로 geodesic 또는 `−d(next)`여서 현재 `100Δd` 설정의 T3 효과를 직접 판정하지 못한다. 분기점의 후보 Q gap/T와 teacher·정책의 양쪽 경로 질량을 함께 기록해야 온도가 왜 작동/실패했는지 알 수 있다. 별개 후보는 T1을 두고 reward scale만100→20으로 줄이는 것이다. 이전 `reward×0.2, T×0.2`는 비율을 함께 유지한 다른 질문이었다. 250k 일시적인 양쪽 진입만으로 성공이라 하지 않는다.
3. **환경 수집 위상 가설:** v3/v4의 256개 환경은 원래 동일 start·700step timeout으로 재시작 시점이 동기화된 기록이 있다. 물리·보상·평가 원점과 정상 episode의 reset state는 유지하면서 **최초 수집 episode의 phase만 분산**하는 대조가 아직 없다. 이는 동시 수집의 상관을 줄일 수 있지만 최초 부분 episode의 수집 분포는 달라지고 효과는 미검증이다. 256→32env 결과만으로 이 실험을 대신할 수 없다.

두 목표를 매 episode 지정하거나, 한 통로에 별도 보너스를 주거나, 평가 시작을 랜덤화하는 방식은 양쪽 궤적을 쉽게 만들 수 있어도 **원래 동일 상태에서 한 정책이 두 경로를 선택하는가**라는 질문을 바꾼다. 그런 결과는 원래 과제의 multimodality 입증으로 사용하면 안 된다. 기존 자료를 읽고 정리했으며 알고리즘·학습 설정·실행 중인 작업은 변경하지 않았다.
