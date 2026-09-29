# 2026-09-21 완료 결과

GMM40 32개(6개 알고리즘×4seed 및 OptiQ 대조군 2개×4seed)가 모두 100k actor updates를 완료했습니다. 원본 archive SHA256, 32개 optimizer update audit, 공통 target/설정을 확인했습니다. cap−3의 W&B 보정도 4개 모두 완료했습니다. 기존 DACER 8개는 1M 최종 actor/critic checkpoint 및 평가 NPZ를 확인하고 로컬 보관했습니다.

## GMM40 baseline: seed0~3, 100k

±는 4시드 sample SD(ddof=1)입니다. Near는 GT 3σ 영역 안의 전체 샘플 비율입니다.

| 방법 | Coverage /40 | Near (%) | MMD² ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|
| optiq_trg | 39.75 ± 0.5 | 68.582 ± 9.69 | 0.0046316 ± 0.00387 | 0.20665 ± 0.0602 |
| sac | 1.75 ± 0.957 | 96.71 ± 3.07 | 0.66011 ± 0.0816 | 0.96375 ± 0.0141 |
| dipo | 36.5 ± 1.73 | 92.007 ± 0.838 | 0.016309 ± 0.00162 | 0.27848 ± 0.013 |
| meow | 24.5 ± 5.45 | 60.005 ± 5.76 | 0.096311 ± 0.0514 | 0.55936 ± 0.104 |
| mfpo | 36 ± 0.816 | 54.822 ± 0.928 | 0.0092031 ± 0.00337 | 0.28593 ± 0.0368 |
| sql | 8.25 ± 2.5 | 91.92 ± 2.61 | 0.20992 ± 0.0522 | 0.79572 ± 0.0618 |

OptiQ는 coverage와 MMD²가 가장 좋았습니다. DIPO는 near 비율이 높지만 놓친 성분이 있고, SAC/SQL은 적은 모드에 집중했습니다. 각 알고리즘은 native architecture/optimizer를 유지했으므로 동일 actor update 수가 동일 Q-query 수·모델 크기·연산량을 뜻하지 않습니다.

![baseline 학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/results/summary/learning_curves.png)
[baseline 전체 24개 최종분포—각 패널 5,000개 표시, 10,000개 원본 보관](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/results/summary/final_distributions.png)

## OptiQ mean 초기화 / sigma 대조군

세 캠페인 모두 100k actor updates, seed 0~3. ±는 시드 간 sample SD(ddof=1). 원본 archive SHA256와 32개 optimizer audit를 검증했습니다.

| 설정 | Coverage /40 | Near 3σ (%) | MMD² ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|
| Original: mean 1e-4, cap/init -1 | 39.75 ± 0.5 | 68.582 ± 9.69 | 0.0046316 ± 0.00387 | 0.20665 ± 0.0602 |
| Mean 1, cap/init -1 | 39.5 ± 0.577 | 71.653 ± 3.44 | 0.0027508 ± 0.0015 | 0.17787 ± 0.0697 |
| Mean 1, cap/init -3 | 33.75 ± 4.99 | 79.91 ± 3.33 | 0.010517 ± 0.00945 | 0.19049 ± 0.0942 |

상한 −1을 유지하고 mean 초기화만 1로 바꾼 설정은 평균 MMD²와 mass TV가 낮아졌으나, 4시드만으로 보편적인 개선을 단정하지 않습니다. Coverage는 거의 같습니다.
mean=1에서 상한·초기 log sigma를 −3으로 낮추면 near 비율은 높지만 coverage가 감소하고 MMD²가 악화됩니다. 상한과 초기값을 함께 바꿨으므로 두 효과를 분리할 수 없습니다.
Target 파일의 원시 checksum은 메타데이터 차이로 다르지만 실제 means/std/weights는 정확히 같습니다. 공통 학습 설정을 config로 대조했습니다.


![OptiQ 비교 학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_final_comparison/learning_curves.png)
![4시드 전체분포](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_final_comparison/all_seeds_distributions.png)

## 기존 DACER: mean-init 1e-4

Ant/Humanoid 각각 seed0~3, 1M step. T=.25, beta=1, DACER=true, mean init=1e-4. 새 mean-init=1 캠페인과 구분합니다.
마지막100k(900000<step<=1000000)의 20회 평가×10episode를 시드 내 평균 후 4시드 평균. ±는 sample SD(ddof=1).

| 환경 | stochastic_z | zero_z |
|---|---:|---:|
| Ant | 5243.2 ± 87.5 | 5503.4 ± 71.8 |
| Humanoid | 5478.4 ± 228.4 | 5199.0 ± 346.9 |

학습곡선은 각 평가의 10episode 평균을 시드 간 평균한 값이며 음영은 시드 간 SD입니다. 두 평가 모두 mu-only이며 stochastic_z는 행동마다 normal latent를 뽑고 DACER 잡음은 평가에 더하지 않습니다.
기존 temperature 결과와는 T도 달라 순수 DACER 효과를 분리할 수 없습니다. Beta 취소 작업은 완료 수에 포함하지 않았습니다.


![DACER 학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/trg_dacer_priority/results/summary/learning_curves.png)
![마지막100k](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/trg_dacer_priority/results/summary/last100k.png)

## 계속 진행 중

새 mean-init 1.0 DACER RL 캠페인은 HalfCheetah seed0,1 및 Ant seed0,1 실행 중, Ant seed2,3 대기입니다. 실패는 발견되지 않았습니다. 중단/재시작/하이퍼파라미터 변경을 하지 않았고, 취소된 beta 등은 재등록하지 않았습니다. 이 캠페인이 완료·보관·보고될 때까지 자동 모니터링을 유지합니다.
