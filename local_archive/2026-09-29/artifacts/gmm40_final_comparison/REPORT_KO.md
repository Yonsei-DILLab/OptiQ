# GMM40 최종 비교

세 캠페인 모두 100k actor updates, seed 0~3. ±는 시드 간 sample SD(ddof=1). 원본 archive SHA256와 32개 optimizer audit를 검증했습니다.

| 설정 | Coverage /40 | Near 3σ (%) | MMD² ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|
| Original: mean 1e-4, cap/init -1 | 39.75 ± 0.5 | 68.582 ± 9.69 | 0.0046316 ± 0.00387 | 0.20665 ± 0.0602 |
| Mean 1, cap/init -1 | 39.5 ± 0.577 | 71.653 ± 3.44 | 0.0027508 ± 0.0015 | 0.17787 ± 0.0697 |
| Mean 1, cap/init -3 | 33.75 ± 4.99 | 79.91 ± 3.33 | 0.010517 ± 0.00945 | 0.19049 ± 0.0942 |

상한 −1을 유지하고 mean 초기화만 1로 바꾼 설정은 평균 MMD²와 mass TV가 낮아졌으나, 4시드만으로 보편적인 개선을 단정하지 않습니다. Coverage는 거의 같습니다.
mean=1에서 상한·초기 log sigma를 −3으로 낮추면 near 비율은 높지만 coverage가 감소하고 MMD²가 악화됩니다. 상한과 초기값을 함께 바꿨으므로 두 효과를 분리할 수 없습니다.
Target 파일의 원시 checksum은 메타데이터 차이로 다르지만 실제 means/std/weights는 정확히 같습니다. 공통 학습 설정을 config로 대조했습니다.

![학습곡선](learning_curves.png)
![seed0 분포](seed0_distributions.png)
![4seed 분포](all_seeds_distributions.png)
