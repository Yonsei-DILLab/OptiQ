256×2에서도 GMM40의 **μ-only near ≥90%, coverage 40/40**을 확인했다. 다만 **100k가 아닌 500k actor updates**에서 달성했다. 학습 seed는 0 하나이며, 독립 latent 재평가는 학습 seed 반복 실험과 구분해야 한다.

| log σ 범위 | 100k near | 500k near | 500k coverage | 새 latent 5묶음 near | 새 latent coverage |
|---|---:|---:|---:|---:|---:|
| [-5, -4] | 75.70% | 90.66% | 40/40 | 90.02–90.92% | 모두 40/40 |
| [-5, -4.5] | 75.19% | 91.05% | 40/40 | 90.72–91.63% | 모두 40/40 |

![100k와 500k μ-only 비교](reproduction_256x2_500k.png)

![학습 추이](reproduction_256x2_500k_curves.png)

두 설정 모두 N=M=64, random Gaussian latent, 256×2, mean-head 초기화 variance scale=16, initial log σ=-4.75, teacher std floor=.05, batch=256, Adam LR=3e-4, T=1, beta=1이다. 각 그림에는 μ 10,000개를 필터링 없이 표시했다. 파란 점은 GT 성분 중 하나의 3σ 경계 안, 주황 점은 모든 경계 밖이다. Coverage는 기존 성분별 최소 점유량 기준을 그대로 사용했다. Near와 coverage가 높아도 분포 전체가 정확히 일치한다는 뜻은 아니며, 모드 사이 연결과 모드 내 형상 차이는 남아 있다.

초기화·σ 설정을 제외한 learner와 sampler 알고리즘은 그대로다. 기존 TRG의 bounded center 구현을 유지했고, 과거 tanh-squashed Gaussian으로 교체하지 않았다. 원본 100k 체크포인트에서 parameters, Adam state, RNG를 그대로 이어 500k 고정 예산을 완료했다. 재개 시 100k μ 배열이 원본과 같았고, save/restore 다음 update가 bit-identical임을 확인했다. 두 최종 optimizer counter 모두 500,000이다.

σ를 GT 모드보다 작게 해도 목표 달성이 가능했다. 상한 -4.5의 physical σ는 40×exp(-4.5)=0.444이며 GT σ≈1.313보다 작다. μ(z)의 분포 자체가 폭을 가질 수 있기 때문이다. 다만 이는 작은 σ가 항상 낫다는 뜻은 아니다. 100k matched 작은-σ 실험에서 512×3 cap -4.5는 91.31%, 40/40이었지만 256×3은 87.45%, 40/40이었다. 이전 256×3 cap -3, initial -3.5는 100k에 95.22%, 40/40이었다. 따라서 이번 256×2는 90% 목표를 재현했으며, 이전 95% 결과나 동일 100k 수렴 속도까지 재현한 것은 아니다.

Full-policy σ-noise 결과는 보조 지표로만 보존한다: cap -4는 89.89%, cap -4.5는 90.88%, coverage는 모두 40/40. 본문 표·그림은 모두 μ-only다. RL 기본 log σ [-5,-1]을 변경하지 않았고 위 범위는 GMM40 ablation 설정이다.

다른 학습 seed의 안정성은 미확인이다. 앞선 256×3 cap -3.5의 4개 학습 seed에서도 일부 seed가 coverage 또는 near 기준에 미달했다. 이번 새 latent 5묶음 검증을 5개 학습 seed 성공으로 해석하면 안 된다.

원본 source: `6aee290c068b9efa58092f6a23c1b4c3cc5f5d8d`.
Resume runner: `a661b78f08e9f63b30f778ce82959ef250f44f58`.
결과 문서·향후 W&B job_type 수정은 사후 commit `3ec2684`로 direct-gmm-trg에 push했다. 완료된 두 run의 legacy 100k job_type만 사후 수정했으며 frozen source와 학습 history는 보존했다.

학습 update/sample AST 및 inherited TRG 파일 4개의 해시가 원래 알고리즘과 같음을 재확인했다. 각 config, 최종 checkpoint, target, μ/full samples, 모든 평가 시점 μ metrics, optimizer audit, independent validation arrays/JSON, source manifest, preflight와 metadata sidecar를 이 폴더에 저장했다. PNG/PDF는 실제 표본으로 생성해 시각 검수했다. 등록된 NM64 탐색은 모두 완료했고 취소된 NM256 실험은 재개하지 않았다.

- [cap -4 W&B](https://wandb.ai/OptiQ/gmm-trg/runs/o2gddxfi)
- [cap -4.5 W&B](https://wandb.ai/OptiQ/gmm-trg/runs/a5fvbz26)
- [상세 수치·설정·검증 JSON](reproduction_256x2_500k_summary.json)
- [PDF](reproduction_256x2_500k.pdf)
- [100k 네 가지 설정 비교](reproduction_256x2.png)
- [작은 σ 6개 설정 비교](small_sigma_comparison.png)
