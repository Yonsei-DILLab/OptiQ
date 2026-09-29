# 기존 DACER 8개 최종 결과

Ant/Humanoid 각각 seed0~3, 1M step. T=.25, beta=1, DACER=true, mean init=1e-4. 새 mean-init=1 캠페인과 구분합니다.
마지막100k(900000<step<=1000000)의 20회 평가×10episode를 시드 내 평균 후 4시드 평균. ±는 sample SD(ddof=1).

| 환경 | stochastic_z | zero_z |
|---|---:|---:|
| Ant | 5243.2 ± 87.5 | 5503.4 ± 71.8 |
| Humanoid | 5478.4 ± 228.4 | 5199.0 ± 346.9 |

학습곡선은 각 평가의 10episode 평균을 시드 간 평균한 값이며 음영은 시드 간 SD입니다. 두 평가 모두 mu-only이며 stochastic_z는 행동마다 normal latent를 뽑고 DACER 잡음은 평가에 더하지 않습니다.
기존 temperature 결과와는 T도 달라 순수 DACER 효과를 분리할 수 없습니다. Beta 취소 작업은 완료 수에 포함하지 않았습니다.

![학습곡선](learning_curves.png)
![마지막100k](last100k.png)
