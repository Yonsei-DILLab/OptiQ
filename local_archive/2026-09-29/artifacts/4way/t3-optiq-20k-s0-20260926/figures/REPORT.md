# 4way OptiQ diagnostic

Four goals are symmetric and every evaluation episode starts at the same origin.
Goal reach frequencies are separate from first-action directional mass.

| step | mode | success | east | west | north | south | no goal | mean return |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -448.809 |
| 1000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -499.540 |
| 2000 | policy | 0.000 | 0 | 0 | 0 | 0 | 40 | -428.231 |
| 2000 | mu_only | 0.000 | 0 | 0 | 0 | 0 | 40 | -482.113 |
| 5000 | policy | 1.000 | 9 | 26 | 3 | 2 | 0 | -150.385 |
| 5000 | mu_only | 1.000 | 6 | 33 | 1 | 0 | 0 | -123.457 |
| 10000 | policy | 1.000 | 12 | 8 | 6 | 14 | 0 | -143.787 |
| 10000 | mu_only | 1.000 | 16 | 8 | 7 | 9 | 0 | -121.822 |
| 20000 | policy | 1.000 | 33 | 35 | 18 | 14 | 0 | -138.170 |
| 20000 | mu_only | 1.000 | 35 | 28 | 18 | 19 | 0 | -117.218 |

One seed and 20,000 transitions are a short diagnostic, not evidence of asymptotic multimodality.
