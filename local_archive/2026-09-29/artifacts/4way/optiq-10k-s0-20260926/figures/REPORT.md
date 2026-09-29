# 4way OptiQ diagnostic

Four goals are symmetric and every evaluation episode starts at the same origin.
Goal reach frequencies are separate from first-action directional mass.

| step | mode | success | east | west | north | south | no goal | mean return |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -448.809 |
| 1000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -499.540 |
| 2000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -432.292 |
| 2000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -461.387 |
| 5000 | policy | 1.000 | 19 | 0 | 19 | 2 | 0 | -128.382 |
| 5000 | mu_only | 1.000 | 21 | 0 | 17 | 2 | 0 | -123.427 |
| 10000 | policy | 1.000 | 71 | 0 | 6 | 23 | 0 | -125.169 |
| 10000 | mu_only | 1.000 | 66 | 0 | 5 | 29 | 0 | -115.631 |

One seed and 10k transitions are a short diagnostic, not evidence of asymptotic multimodality.
