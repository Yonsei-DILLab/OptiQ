# AntMaze 다중 목표·경로 비교

완료 4/4. 학습 예산: v1=100k online interactions.

주 그림은 동일한 전체 시뮬레이터 초기 상태에서 정책을 직접 샘플링한100회 rollout입니다. 각 칸은 한 training seed의 정책이며 시드를 섞지 않았습니다.

| 미로 | 방법 | 완료seed | 성공률 | 유효 경로 수 | 최다 경로 비중 |
|---|---|---:|---:|---:|---:|
| v1 | optiq | 1/1 | 0.000 | 0.000 | N/A |
| v1 | sac | 1/1 | 0.000 | 0.000 | N/A |
| v1 | mfpo | 1/1 | 0.000 | 0.000 | N/A |
| v1 | meow | 1/1 | 0.000 | 0.000 | N/A |

유효 경로 수는 성공 rollout의 경로 범주 엔트로피를 exp한 값입니다. 성공이 없으면0으로 표시하며 mode collapse라고 단정하지 않습니다.
성공 경로가 없는 정책의 최다 경로 비중은 N/A입니다. 이 값의 평균/편차는 성공 경로가 관측된 training seed만으로 계산하고 해당 seed 수를 표시합니다.
경로는 사전에 정의한 통로 횡단으로 분류했습니다. 모든 가능한 homotopy class나 action 분포의 multimodality를 증명하는 값은 아닙니다.
v4의 원래 초기 상태 분포는 고정되어 natural/fixed 평가 조건이 같습니다. 두 결과를 독립된200회로 합치지 않습니다.
OptiQ 주 그림은 learned sigma를 포함한 정책 샘플입니다. DACER 외부 행동잡음은 제외하며 μ-only 그림은 별도 보조 자료입니다.
원본 저속 Ant/미로를 MuJoCo3로 이식하고 명시적인 최근접목표 거리 보상을 적용했습니다. 미공개 MFPO AntMaze 설정의 정확한 재현으로 해석하지 않습니다.
![Seed0](seed0-overview.png)

## 이번 파일럿 설정과 해석

DDiffPG NovelD와 환경256개, batch4096, 수집 round당8updates를 적용했습니다. Warmup8192를 포함한 정확100k interactions, 실제 learner/RND 업데이트2872회입니다. 각 방법 고유 모델/LR/tau는 유지했습니다. 보상은 dense 최근접목표 거리 penalty이며 평가에는 NovelD와 DACER 외부 잡음을 넣지 않았습니다.

이전 dense-only100k와 비교하면 NovelD, batch, 병렬 수집, 업데이트 빈도와 warmup이 함께 달라 순수 intrinsic reward ablation이 아닙니다. 각 방법 training seed0 하나의 결과이므로 seed간 우위는 판정하지 않습니다.

| 방법 | 본실험 시간(평가 포함, 분) | fixed 성공/100 | natural 성공/100 | fixed 평균 최단거리 |
|---|---:|---:|---:|---:|
| optiq | 3.44 | 0 | 0 | 6.610 |
| sac | 2.35 | 0 | 0 | 6.818 |
| mfpo | 2.88 | 0 | 0 | 7.876 |
| meow | 3.04 | 0 | 0 | 7.092 |

마지막 학습 minibatch의 평균 intrinsic reward는0.0074~0.0094, 환경 보상 평균은약-8.0~-8.3이었습니다. 원본계수0.01을그대로사용했으며, 평균보상절댓값대비약0.09~0.12%입니다. 이크기비교만으로실패원인을확정할수없습니다.
