# GMM40 100k · 4-seed 비교

소스: `87d5d8ff210569ace7e59bd8a53ad02141b67f0a`

고정 Q = 원본 GMM40 log density, T=1. 각 방법 seed 0~3, 100,000 actor updates, 평가당 10,000 samples.
표의 ±는 학습 시드 간 표준편차(ddof=1)입니다. Target 샘플은 평가에서만 사용합니다.
Supplement: full stochastic policies
그림·표·학습곡선은 동일한 평가 모드를 사용합니다. μ-only에서도 latent는 원래 정책의 prior대로 샘플링합니다.

| Method | Coverage /40 | Within 3σ | MMD² ↓ | SW ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|---:|
| OptiQ Direct GMM/TRG | 39.75 ± 0.5 | 0.68583 ± 0.0969 | 0.0046316 ± 0.00387 | 2.4281 ± 0.836 | 0.20665 ± 0.0602 |
| SAC | 1.75 ± 0.957 | 0.9671 ± 0.0307 | 0.66011 ± 0.0816 | 37.887 ± 2.92 | 0.96375 ± 0.0141 |
| DIPO | 36.5 ± 1.73 | 0.92007 ± 0.00838 | 0.016309 ± 0.00162 | 4.8944 ± 0.413 | 0.27848 ± 0.013 |
| MEOW | 24.5 ± 5.45 | 0.60005 ± 0.0576 | 0.096311 ± 0.0514 | 14.988 ± 7.28 | 0.55936 ± 0.104 |
| MFPO | 36 ± 0.816 | 0.54822 ± 0.00928 | 0.0092031 ± 0.00337 | 3.3474 ± 0.835 | 0.28593 ± 0.0368 |
| SQL (JAX SVGD) | 8.25 ± 2.5 | 0.9192 ± 0.0261 | 0.20992 ± 0.0522 | 17.876 ± 0.578 | 0.79572 ± 0.0618 |

OptiQ는 box-truncated Direct GMM NLL, N=64, M=64, random latent, 256×2, log σ 하한 -5.0, -1.0, mean-head init scale 0.0001입니다.
다른 baseline은 기존 native architecture와 optimizer를 유지했습니다. 모든 방법의 batch는 256입니다.
동일 update 수 비교이며 Q 질의량과 계산량은 다릅니다. `per_seed.csv`의 Q_evaluations, train_seconds, parameters를 함께 보세요.
GMM component coverage는 density의 실제 local maxima 개수와 같지 않습니다.

![학습곡선](learning_curves.png)
![최종지표](final_metrics.png)
![최종분포](final_distributions.png)
