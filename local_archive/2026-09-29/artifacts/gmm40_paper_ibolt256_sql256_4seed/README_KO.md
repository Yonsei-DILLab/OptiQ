# GMM40 논문 그림: SQL K=256

- 그림은 이전과 같은 training seed 2의 GT, SAC, SQL, MFPO, DIPO, iBOLT 결과다. SQL만 SVGD K=256(고정 128, 갱신 128)으로 교체했다.
- 표는 모든 알고리즘의 training seed 0–3, 각 100k actor updates 결과를 각각 검증한 뒤 지표별 평균 ± seed 간 sample SD로 집계했다. Ground truth는 공통 고정 참조다.
- 모든 패널은 해당 정책의 native full-policy 샘플 10,000개이며 iBOLT는 random latent와 conditional sigma를 포함한다. 점 면적 2.10 pt², alpha 0.2, 파랑, 같은 GT log-density 등고선을 사용한다.
- MMD는 각 시드의 2,048개 샘플과 같은 독립 GT draw로 계산한 MMD²의 제곱근이다. 표는 시드별 MMD의 평균이며 `sqrt(mean(MMD²))`가 아니다.
- 모든 20개 학습 run의 완료·optimizer audit, 표본 모양·유한값·범위, GT 정의, 3σ mode-count, MMD²를 원자료에서 재검증했다. 경로와 SHA256은 `results.json` 및 `per_seed.csv`에 보관했다.
- 학습 예산은 같지만 SQL K=256은 기존 K=16보다 Q query가 16배다. iBOLT·baseline은 알고리즘 고유 아키텍처를 유지했다.
