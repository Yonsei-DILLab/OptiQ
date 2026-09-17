# 1D non-stationary Q: OT와 Direct GMM의 추적 비교

이 보고서는 **동일하게 변하는 Q를 누가 더 잘 추적하는가**, 그리고 **실제 actor–critic에서는 어떤 결과가 생기는가**를 나누어 설명한다. 결과가 없는 칸은 미완료로 남긴다.

## 설정과 그림 읽는 법

| 항목 | 값 |
|---|---|
| Actor | v5 256×2 GELU, conditional squashed Gaussian |
| 비교 | GMM learned σ; Exact/Sinkhorn 각각 learned σ, fixed σ=0.5, fixed σ=0.1 |
| N×M | 16×64 / 256×16384 / 1024×4096 / 2048×2048 |
| Seeds / temperature | 0,1,2,3 / τ=0.25 고정 |
| 학습 | Adam3e−4, clipping 없음; 각 최종 궤적35K actor updates |
| Actor density | **32768개 latent와 독립 Gaussian noise로 실제 action을 생성한512-bin histogram** |
| Smoothing | 없음. Actor 곡선에 Gaussian CDF 적분이나 KDE를 사용하지 않음 |
| Teacher | 현재 actor의 conditional-mixture proposal에서 후보를 뽑고 Q/.25−log q로 재가중 |
| Critic 학습 | Source/closed만 수행. 5K uniform warmup+35K 학습, reward-only TD |

**N은 별도 network 수가 아니다.** 하나의 actor에 입력한 N개 latent가 만드는 N개의 conditional Gaussian이다. Learned sigma는 평균과 분산을 함께 학습하고, fixed sigma는 행동 생성과 학습 모두 같은 고정값을 사용한다. Fixed0.5는 좁은 mode의 표현 자체가 어려울 수 있다.

**Histogram TV**는 모든 bin에서 목표와 actor의 확률 질량 차이 절댓값을 더한 뒤 절반을 취한다. **Basin-mass TV**는 target peak 사이 valley로 구간을 나눈 뒤 같은 계산을 한다. 목표가 (0.5,0.5), actor가 (0.8,0.2)이면 basin TV는0.3이다. Basin TV가 낮아도 mode 사이 빈 공간을 넓은 Gaussian으로 채울 수 있으므로 density와 histogram TV를 먼저 본다.

점선 target은 analytic Q에서는 정규화된 정답이며, learned Q에서는 저장된 critic을 독립 격자로 평가한 수치 reference다. Closed actor–critic에서는 방법마다 critic이 다르므로 **각 방법 자신의 target**과 짝지어 보여준다. 여러 seed를 평균한 target은 개별 seed의 단일 분포가 아니다.

**Heatmap:** OT 그림은 실제 loss에 들어간 R/N이다. 유한 반복 Sinkhorn의 원래 P는 raw 파일에 따로 보관하고 marginal 오차도 기록한다. GMM에는 OT plan이 없으므로 w_jγ_ij를 effective assignment로 그린다. 원본 sampling 순서와 action 정렬본은 같은 행렬이며 같은 색 범위를 사용한다. 큰 행렬은 표시할 때 연속 block 평균을 사용하고, 전체 원본과 정렬 index는 보관한다.

Mean·sigma 궤적은 고정 평가 latent로 추적한다. 매 update 새로 뽑는 training row를 영구적인 component identity로 연결하지 않는다. Proposal/weighted teacher/actor를 따로 비교하며, actor–teacher TV가 teacher 오류와 actor fitting 오류의 정확한 가법 분해라고 주장하지 않는다.

공통 prefix는20K까지, mass/split은 이를 복사해 각각35K까지 진행한다. Replay는 실제 v5 critic의 Q(0,a) 궤적을 동일하게 제공하며, closed는 각 방법이 자기 critic을 학습한다. 최종 독립 평가 histogram의 mean±sample SD를 제공한다. 예시 trajectory와 assignment heatmap은 seed0/N16을 사용하고, teacher·actor 비교와 고정 latent 그림은 네 크기 모두 seed0을 사용한다. 예시 선택은 결과를 보기 전에 고정했다.

**Teacher histogram의 표본 수는 M개**다. 특히 M=64에서는512개 bin의 세부 density를 맞출 수 없으므로 teacher histogram TV가 표본 수만으로도 커진다. 동일 크기 내 방법 비교, basin 질량, 여러 시점의 추세를 함께 확인한다. Actor의32768개 histogram과 teacher의M개 histogram을 같은 표본 정밀도로 해석하지 않는다.

Mean·sigma 그림은 사전 지정한16개 평가 latent index를 사용한다. 왼쪽의 tanh(mu)는 conditional distribution의 action-space 중앙값이며 action의 평균과는 다르다. 오른쪽 sigma는 pre-tanh 표준편차다. Component 지표의 usage ESS/N은 assignment row mass의 균등성으로, OT가 높다고 실제 여러 행동 mode의 분업이 보장되는 것은 아니다. Seed 평균 곡선은 그려지는 시점마다 구성 seed가 바뀌지 않도록 공통 관측 시점만 사용한다.
