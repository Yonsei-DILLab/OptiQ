# GMM40 논문 피겨 · iBOLT

Ground truth → SAC → SQL → MFPO → DIPO → iBOLT 순서입니다. 본문은 2×3이며 1×6 대안도 저장했습니다.

- 본문: `gmm40_main_seed0.pdf` / `.svg` / `.png` (seed 0, 모든 표본 10,000개)
- 본문 표: `gmm40_table.tex` (학습 seed 0~3 평균 ± 표본 표준편차)
- seed 0 그림과 일대일 대응하는 표: `gmm40_table_seed0.tex`
- 캡션과 삽입 코드: `gmm40_figure.tex`
- 모든 시드 보조 그림: `gmm40_all_four_seeds.pdf` / `.png`
- iBOLT σ 포함 보조 그림: `gmm40_ibolt_full_policy_supplement.pdf` / `.png`

전부 100k actor updates. iBOLT는 사용자 선택의 GMM40 튜닝 설정(256×3, N=M64, mean-head variance scale16, logσ[-5,-3.5], 초기-4, teacher floor.05)입니다. 기본 256×2 / logσ[-5,-1] 결과 또는 500k 결과와 섞지 않았습니다. baseline은 native architecture/optimizer를 유지하며 계산량·Q 질의량이 같다는 비교는 아닙니다.

그림과 표 모두 iBOLT는 fresh Gaussian z의 μ-only, baseline은 native generator output입니다. 모드 간 연결을 포함한 모든 표본을 동일한 점 크기·투명도로 표시하며 근접 여부로 걸러내지 않았습니다. 시드를 섞거나 모델별로 좋은 시드를 고르지 않았습니다. μ-only 결과를 full-policy density 성능으로 해석하지 마세요. σ 포함 결과도 별도로 보존했습니다.

MMD는 기존 unbiased MMD² 추정값을 시드별 제곱근 변환한 값입니다. 원시 표본에서 재계산해 원래 저장값과 1e-10 이내 일치를 확인했습니다. 5개 RBF bandwidth(1,2,5,10,20) 평균, 원래 순서 첫 2048개 표본과 공통 reference를 사용합니다. MMD²도 JSON에 보존합니다. Ground truth는 기존 target sampler의 bounded=True, seed20260917로 재생성했습니다.

Coverage는 10,000개 전체에서 최근접 성분의 3σ 이내 표본 수가 max(10, reference 성분 점유 수×.1) 이상인 성분 수입니다. 분포의 엄밀한 국소 극대점 개수는 아닙니다. target means/std/weights/scale/bounded_mass와 20개 optimizer audit를 확인했습니다.

| Algorithm | MMD ↓ | Coverage /40 ↑ | MMD² (archived) |
|---|---:|---:|---:|
| iBOLT | 0.0614 ± 0.0208 | 38.75 ± 2.50 | 0.004090 ± 0.002853 |
| SAC | 0.8112 ± 0.0519 | 1.75 ± 0.96 | 0.660109 ± 0.081612 |
| SQL | 0.4552 ± 0.0606 | 8.25 ± 2.50 | 0.209922 ± 0.052237 |
| MFPO | 0.0949 ± 0.0164 | 36.00 ± 0.82 | 0.009203 ± 0.003373 |
| DIPO | 0.1276 ± 0.0064 | 36.50 ± 1.73 | 0.016309 ± 0.001620 |

iBOLT 각 seed coverage: 40, 35, 40, 40. 본문 seed0의 40/40을 4시드 전부의 결과로 일반화하지 않습니다.

보존된 OptiQ run 이름/소스/설정은 provenance를 위해 유지하고 논문 및 새 보고의 명칭만 iBOLT로 사용합니다.

재생성: `/tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_figure.py`
