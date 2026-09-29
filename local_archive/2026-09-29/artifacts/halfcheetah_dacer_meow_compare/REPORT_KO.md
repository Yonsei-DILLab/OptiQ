# HalfCheetah: OptiQ GMM-TRG T=0.25, beta=1, DACER on vs baseline MEOW

W&B의 unsampled scan_history로 계산. OptiQ seed 4의 최종 평가가 875k인 시점의 스냅샷이다.

완료된 동일 seed 0,1,2,3 비교. 마지막100k는 900000 < step <= 1000000, seed별20평가×10episode를 평균한 뒤 seed 평균과 표본 표준편차를 계산했다.

| 평가 | 마지막100k 평균 ± seed SD | 1M 평가 평균 | 직전100k 대비 변화 |
|---|---:|---:|---:|
| OptiQ stochastic_z | 10,316.13 ± 852.10 | 10,431.27 | +265.64 |
| OptiQ zero_z | 10,522.34 ± 880.41 | 10,414.93 | +266.85 |
| MEOW deterministic | 11,407.23 ± 376.27 | 11,591.97 | +171.28 |

OptiQ stochastic_z가 MEOW보다 1,091.10점(9.56%) 낮다. 두 방법 모두 후반 상승하며, 마지막100k의 앞50k→뒤50k 변화는 OptiQ +105.52, MEOW +142.74다.

OptiQ zero_z 평균도 MEOW보다 낮다. 평가는 OptiQ stochastic_z/zero_z와 MEOW deterministic으로 구분한다. 서로 다른 평가 정책과 알고리즘 설정을 사용하므로 통제된 단일요인 비교로 해석하지 않는다.

5개 seed 전체의 공통 구간 775000 < step <= 875000 평균:

- OptiQ stochastic_z: 9,967.72 ± 743.61
- OptiQ zero_z: 10,177.92 ± 735.03
- MEOW deterministic: 11,148.97 ± 305.18

## 마지막100k seed별 평균

| seed | OptiQ stochastic_z | OptiQ zero_z | MEOW deterministic |
|---|---:|---:|---:|
| 0 | 9,974.36 | 10,134.81 | 11,110.81 |
| 1 | 9,583.39 | 9,731.79 | 11,217.14 |
| 2 | 11,541.58 | 11,765.72 | 11,348.47 |
| 3 | 10,165.18 | 10,457.03 | 11,952.48 |

## 원본 run

- [baseline seed 0](https://wandb.ai/OptiQ/baseline/runs/ouahytpa) (finished)
- [baseline seed 1](https://wandb.ai/OptiQ/baseline/runs/do6ns7s1) (finished)
- [baseline seed 2](https://wandb.ai/OptiQ/baseline/runs/7cy19njx) (finished)
- [baseline seed 3](https://wandb.ai/OptiQ/baseline/runs/2mfrwjex) (finished)
- [baseline seed 4](https://wandb.ai/OptiQ/baseline/runs/6hv8h2yf) (finished)
- [gmm-trg seed 0](https://wandb.ai/OptiQ/gmm-trg/runs/b278f114e632) (finished)
- [gmm-trg seed 1](https://wandb.ai/OptiQ/gmm-trg/runs/7d6785ab8aa3) (finished)
- [gmm-trg seed 2](https://wandb.ai/OptiQ/gmm-trg/runs/e2605d83f9d6) (finished)
- [gmm-trg seed 3](https://wandb.ai/OptiQ/gmm-trg/runs/08ff8a4dc94f) (finished)
- [gmm-trg seed 4](https://wandb.ai/OptiQ/gmm-trg/runs/894a864c9a92) (running)
