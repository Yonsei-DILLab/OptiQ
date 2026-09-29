# GMM40: 99% and 95% target highest-density regions

기존 v5의 100k / seed0 / full-policy 샘플을 그대로 사용한 색상 비교입니다.
Ground truth, SAC, SQL, MFPO, DIPO, iBOLT 순서를 유지했습니다.
iBOLT conditional sigma, SAC sigma 및 각 baseline의 native stochastic 생성 결과를 유지했습니다.

99%·95%는 GT 확률질량 기준입니다. 각 알고리즘의 빨간 표본 비율을 미리 고정하지 않습니다.
GT에서 독립적으로 생성한 1,000,000개 log density의 하위 1%/5% 분위수로 임계값을 정했습니다.
별도의 독립 GT 1,000,000개로 포함 확률을 검증했습니다. 같은 임계값을 모든 방법/시드에 적용합니다.
밀도는 40개 성분의 가중합이며, 기존 평가와 동일한 action box 조건부 GT를 사용합니다.
파랑: log p_GT(x) >= threshold. 빨강: log p_GT(x) < threshold.

| HDR | Density threshold | 독립 GT 포함률 |
|---|---:|---:|
| 99% | 3.591841044e-05 | 98.9885% |
| 95% | 0.000160562055 | 94.9867% |

| seed0 | 기존 3sigma 밖 | 99% HDR 밖 | 95% HDR 밖 |
|---|---:|---:|---:|
| Ground truth | 0.76% | 1.01% | 5.23% |
| SAC | 2.11% | 2.60% | 6.57% |
| SQL | 10.65% | 11.56% | 22.63% |
| MFPO | 44.74% | 46.62% | 60.37% |
| DIPO | 8.08% | 8.74% | 15.21% |
| iBOLT | 10.52% | 11.26% | 20.96% |

MMD 및 기존 mode coverage는 다시 정의하거나 변경하지 않았습니다.
GT에서도 HDR 밖 표본이 약 1%/5% 발생하므로 빨간 점을 확정적인 OOD로 해석하지 않습니다.
이전 3sigma 그림·학습 결과·원자료는 보존했습니다.

재생성: /tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_density_hdr_versions.py
