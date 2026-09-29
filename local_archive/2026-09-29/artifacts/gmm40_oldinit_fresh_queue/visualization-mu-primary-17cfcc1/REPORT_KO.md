# GMM40 100k · 4-seed 비교

소스: `85aee3ecae09847954cb999680f854302f148db6`

고정 Q = 원본 GMM40 log density, T=1. 각 방법 seed 0~3, 100,000 actor updates, 평가당 10,000 samples.
표의 ±는 학습 시드 간 표준편차(ddof=1)입니다. Target 샘플은 평가에서만 사용합니다.
Primary: OptiQ μ only; other methods native outputs
그림·표·학습곡선은 동일한 평가 모드를 사용합니다. μ-only에서도 latent는 원래 정책의 prior대로 샘플링합니다.

| Method | Coverage /40 | Within 3σ | MMD² ↓ | SW ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|---:|
| OptiQ Direct GMM/TRG | 32.25 ± 6.24 | 0.73388 ± 0.103 | 0.017226 ± 0.0144 | 4.0019 ± 2.34 | 0.34849 ± 0.118 |

OptiQ는 box-truncated Direct GMM NLL, N=64, M=64, random latent, 256×2, log σ 하한 -5.0, 1.0, mean-head init scale 1.0입니다.
다른 baseline은 기존 native architecture와 optimizer를 유지했습니다. 모든 방법의 batch는 256입니다.
동일 update 수 비교이며 Q 질의량과 계산량은 다릅니다. `per_seed.csv`의 Q_evaluations, train_seconds, parameters를 함께 보세요.
GMM component coverage는 density의 실제 local maxima 개수와 같지 않습니다.

![학습곡선](learning_curves.png)
![최종지표](final_metrics.png)
![최종분포](final_distributions.png)
