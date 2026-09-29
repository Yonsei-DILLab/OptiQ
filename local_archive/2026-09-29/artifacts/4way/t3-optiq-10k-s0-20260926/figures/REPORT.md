# 4way OptiQ diagnostic

Four goals are symmetric and every evaluation episode starts at the same origin.
Goal reach frequencies are separate from first-action directional mass.

| step | mode | success | east | west | north | south | no goal | mean return |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -448.809 |
| 1000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -499.540 |
| 2000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -428.232 |
| 2000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -482.113 |
| 5000 | policy | 1.000 | 9 | 26 | 3 | 2 | 0 | -151.322 |
| 5000 | mu_only | 1.000 | 6 | 33 | 1 | 0 | 0 | -124.215 |
| 10000 | policy | 1.000 | 43 | 20 | 16 | 21 | 0 | -142.216 |
| 10000 | mu_only | 1.000 | 46 | 17 | 20 | 17 | 0 | -121.230 |

One seed and 10,000 transitions are a short diagnostic, not evidence of asymptotic multimodality.
