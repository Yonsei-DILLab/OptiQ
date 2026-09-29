# Gradient clipping 읽기 전용 조사

코드, 설정, 실행 프로세스 변경 없음. 분석 파일만 별도로 저장.

대상: Ant fixed64 T=.25 4 seeds, Humanoid fixed64 T=.1 4 seeds. Actor/critic 256x2, batch256, UTD1, Adam lr3e-4, beta1=.9, beta2=.999, clipping 없음. 최신 로그 시점은 Ant375–380k, Humanoid373–464k 환경 step이며 실험은 진행 중이다. 이 수치는 continuous latent 기본 정책에 대한 측정이 아니다.

## 측정 방법과 한계

현재 CSV/학습 코드에는 raw gradient norm 기록이 없다. 저장된 Adam `nu`를 읽어 `sqrt(sum(nu)/(1-beta2**count))`를 계산했다. 이는 minibatch gradient global L2 norm 제곱의 bias-corrected 지수이동평균에 대한 제곱근이다. 원시 순간 gradient norm, max norm, percentile 또는 clipping 빈도와 같지 않다. beta2=.999는 대략 최근1000 업데이트 규모의 기억을 가진다. 첫 update checkpoint와 50k 간격 checkpoint를 읽었고 아래 범위/그림은 50k 이상만 사용했다. Actor/critic 138개 checkpoint 파일 읽기 성공, 오류0; 표/그림에는 초기 update16개를 제외한122개 사용.

Loss나 reward 변화만으로 gradient 폭발 또는 clipping으로 인한 성능 개선을 판정하지 않는다. 두 환경 간 온도, 관측/action 차원, 학습 진행도가 달라 norm 차이의 원인을 하나로 단정할 수 없다.

## 측정 범위와 제안

| 환경 | 대상 | 체크포인트 EMA RMS 범위 | 첫 탐색 후보 | 보수적 첫 후보 |
|---|---|---:|---:|---:|
|Ant|Actor|2.05–4.74|5, 10|10|
|Ant|Critic (두 Q 합산)|48.1–746.7|500, 1000|1000|
|Humanoid|Actor|17.2–777.2|50, 100|100|
|Humanoid|Critic (두 Q 합산)|3918.6–17729.1|10000, 20000|20000|

후보는 큰 gradient를 제한하기 위한 초기 실험 범위이며 최적성/효과 검증 없음. 먼저 actor-only 후보를 no-clip 대조군과 비교하는 것이 진단 목적상 명확하다. Critic은 norm이 크다는 사실만으로 잘못됐다고 볼 수 없으므로 우선 no-clip 유지, 별도 실험에서 위 후보를 비교하는 접근을 권한다.

현재 `alg.optimizer.ac_grad_norm`은 actor와 critic에 동일 임계값을 전달한다. 위 표의 분리 설정을 현재 공통 옵션 하나로 구현할 수 없다. 기존 v3/v4 값2.0을 이번 두 정책의 적정값으로 취급할 근거가 없으며, 특히 critic 규모와 맞지 않는다. 임계값50이나100 하나를 두 optimizer에 동시에 넣으면 actor clipping 효과와 critic clipping 효과가 섞인다.

## 관찰

Humanoid seed1 actor RMS: 200k39.74 ->250k186.67 ->300k519.89 ->350k777.23. 350k의 mu 출력층 RMS766.27로 전체 squared norm의 97.20%. seed2도 400k66.10 ->450k144.84 상승. 다른 두 seed의 후기 actor RMS는 약43–60. 이 구간에도 로그 actor loss는 대체로22–23 수준이며 nonfinite는 관측되지 않았다. 따라서 loss 스칼라만으로 큰 gradient 통계를 놓칠 수 있다. 상승과 성능 변동의 인과관계는 확인되지 않았다.

Humanoid 후기 teacher ESS는64후보 중 중앙값 약1.9–2.0이다. clipping은 gradient 크기 제한이며 teacher weight 집중 자체를 직접 바꾸지 않는다.

현재 구현은 Optax `clip_by_global_norm`을 Adam 앞에 둔다. Actor는 전체 actor parameter tree, critic은 twin-Q를 포함한 전체 critic tree 기준이다. Adam 전 clipping 임계값은 parameter update norm 또는 learning rate와 같은 값이 아니다.

## 근거 파일/자료

- 원시 분석: `ant.json`, `humanoid.json` (checkpoint 경로, optimizer config, 시점별 통계, 평가 기록 포함)
- 요약: `summary.json`
- 그림: `gradient_scale.png`, `gradient_scale.pdf`; 회색은 미검증 후보 범위.
- 현재 코드: `optiq_dime/optimizers.py`, `optiq_dime/policy.py`; v3 clipping2.0, v5 clippingnull.
- [Optax clipping 및 global norm 정의](https://optax.readthedocs.io/en/stable/api/transformations.html)
- [Optax clipping 후 Adam 적용 예시](https://optax.readthedocs.io/en/stable/getting_started.html)
- [Adam 원 논문](https://arxiv.org/abs/1412.6980)
