# GMM40 μ-only 최종 비교

새 old-init/fresh 캠페인 seed0~3이 모두 100k actor updates를 완료했습니다. Config·평가·checkpoint를 내려받아 archive SHA256 및 optimizer update audit를 검증했습니다. 학습 소스는 85aee3ecae09847954cb999680f854302f148db6입니다.

아래 그림·지표·학습곡선은 모두 conditional sigma noise를 제거한 μ-only 평가입니다. Latent는 학습 설정대로 fresh normal을 사용합니다. 이전 답변의 full-policy 수치와 직접 혼합하지 않습니다. Full-policy 원본은 별도 보조 결과로 보존했습니다.

| 설정 | Coverage /40 | Near 3σ (%) | MMD² ↓ | Mass TV ↓ |
|---|---:|---:|---:|---:|
| 기존 mean=1e-4, cap/init=−1 | 31.25 ± 3.59 | 73.2 ± 8.41 | 0.013787 ± 0.00269 | 0.3338 ± 0.0495 |
| mean=1, cap/init=−1 | 31.75 ± 3.95 | 74.71 ± 5.56 | 0.012172 ± 0.00533 | 0.34152 ± 0.0608 |
| mean=1, cap/init=−3 | 33.5 ± 4.65 | 89.007 ± 2.91 | 0.01241 ± 0.0107 | 0.24551 ± 0.0813 |
| 새 old-init: cap=+1, init σ=.5, teacher floor=.05 | 32.25 ± 6.24 | 73.388 ± 10.3 | 0.017226 ± 0.0144 | 0.34849 ± 0.118 |

±는 학습 시드 4개의 sample SD(ddof=1)입니다. Mass TV는 GT 3σ 안에 들어온 샘플로 조건부 계산되므로 near 비율과 함께 해석해야 합니다.
새 설정은 평균 near·MMD·mass TV에서 기존 mean=1, cap−1 대조군을 개선하지 못했습니다. 4시드 차이가 크며 보편적인 성능 우열을 확정하지 않습니다. cap−3 설정은 μ-only near와 mass TV에서 더 좋지만, 평균 6.5개 성분을 놓칩니다.

## 새 old-init 4시드

| Seed | Coverage /40 | Near (%) | MMD² | Mass TV |
|---|---:|---:|---:|---:|
| 0 | 26 | 63.82 | 0.03246 | 0.4797 |
| 1 | 28 | 65.32 | 0.02650 | 0.4139 |
| 2 | 36 | 80.31 | 0.00476 | 0.2731 |
| 3 | 39 | 84.10 | 0.00518 | 0.2273 |

![새 설정 μ-only 4시드](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_mu_primary_comparison/oldinit_four_seeds.png)
![μ-only 학습곡선 비교](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_mu_primary_comparison/learning_curves.png)

μ-only에도 모드 사이를 잇는 점들과 경계 근처에 모인 점들이 남습니다. 평균 26.61%가 모든 GT 3σ 영역 밖에 있으므로, 이 현상을 conditional sigma sampling noise만으로 설명할 수 없습니다.

## Sigma 상한 진단

최종 checkpoint에서 CPU로 별도 10,000 normal latent를 계산했습니다. 아래 비율은 latent×action좌표 기준이며, 저장된 GPU μ 샘플과 일대일 원인 연결을 주장하지 않습니다.

| Seed | raw log σ > +1 좌표 | raw log σ 최댓값 |
|---|---:|---:|
| 0 | 2.040% | 47.742 |
| 1 | 2.235% | 19.665 |
| 2 | 1.930% | 28.807 |
| 3 | 0.000% | 0.280 |

Seed0~2에는 일부 상한 포화가 남았고 seed3은 이 진단 표본에서 관찰되지 않았습니다. σ 상한 exp(1)은 normalized action의 Gaussian parameter이며 box truncation 이후의 실제 표준편차와는 다릅니다. CPU forward와 저장 GPU μ 간 최대 차이는 물리좌표에서 약 0.54~0.95여서 작은 수치 차이까지 동일한 샘플로 취급하지 않습니다.
상한·초기 sigma·teacher floor가 함께 변했으며, 기존 tanh-policy와 현재 box-truncated policy는 좌표계/분포가 다릅니다. 따라서 이전 정책의 정확한 재현 또는 개별 설정의 단독 효과로 해석하지 않습니다.

## 보관·검증

보고 소스는 17cfcc1ea82618e838fb692d254cbfceeefdfa9b입니다. 4개 완료 GMM40 캠페인 모두 별도 visualization-mu-primary-17cfcc1 폴더에 보고서를 만들고 중간 μ metric 16회×4seed도 로컬 보관했습니다. 각 PROVENANCE.json에 학습/보고 소스 commit과 원본 archive·result·summary의 전후 동일 SHA256을 기록했습니다. 기존 학습 artifact를 덮어쓰지 않았습니다.

- [6개 baseline 비교: OptiQ μ-only, 나머지 native generator](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/visualization-mu-primary-17cfcc1/REPORT_KO.md): [주 분포](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/visualization-mu-primary-17cfcc1/final_distributions.png), [학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/visualization-mu-primary-17cfcc1/learning_curves.png), [full-policy 보조 보고](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_5090_queue/visualization-mu-primary-17cfcc1/full_policy/REPORT_KO.md).
- [mean=1/cap−1](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm1_mean1_queue/visualization-mu-primary-17cfcc1/REPORT_KO.md): [주 분포](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm1_mean1_queue/visualization-mu-primary-17cfcc1/final_distributions.png), [학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm1_mean1_queue/visualization-mu-primary-17cfcc1/learning_curves.png), [full-policy 보조 보고](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm1_mean1_queue/visualization-mu-primary-17cfcc1/full_policy/REPORT_KO.md).
- [mean=1/cap−3](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm3_queue/visualization-mu-primary-17cfcc1/REPORT_KO.md): [주 분포](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm3_queue/visualization-mu-primary-17cfcc1/final_distributions.png), [학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm3_queue/visualization-mu-primary-17cfcc1/learning_curves.png), [full-policy 보조 보고](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_capm3_queue/visualization-mu-primary-17cfcc1/full_policy/REPORT_KO.md).
- [old-init/fresh](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_oldinit_fresh_queue/visualization-mu-primary-17cfcc1/REPORT_KO.md): [주 분포](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_oldinit_fresh_queue/visualization-mu-primary-17cfcc1/final_distributions.png), [학습곡선](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_oldinit_fresh_queue/visualization-mu-primary-17cfcc1/learning_curves.png), [full-policy 보조 보고](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/gmm40_oldinit_fresh_queue/visualization-mu-primary-17cfcc1/full_policy/REPORT_KO.md).

Baseline은 native architecture/optimizer를 유지했으므로 actor update 수가 같아도 모델 크기·Q-query 수·연산량은 다릅니다.

새 mean-init=1 DACER RL 캠페인은 4개 실행·Ant seed2,3 대기이며 실패가 없습니다. 기존 DACER와 나머지 GMM40는 완료 상태입니다. Beta 등 취소 작업을 재개하지 않았고 학습/평가/기존 W&B 지표를 변경하지 않았습니다. RL 6개 완료·보관·보고 전까지 자동 모니터링을 유지합니다.
