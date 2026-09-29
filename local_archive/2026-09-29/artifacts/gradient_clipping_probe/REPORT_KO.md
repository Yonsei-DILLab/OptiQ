# 중지 후 gradient clipping 짧은 진단

Ant4/Humanoid4 학습 worker 및 Ant live-sync worker 중지. 체크포인트/로컬 로그 보존. 자동 재시작 없음. 진단 worker8개 모두 완료 후 종료. 저장소 학습 코드·설정 수정 없음; 독립 진단 스크립트만 실행.

## 방법

원래 frozen source의 update_actor/update_critic를 호출한다. optimizer를 진단 프로세스 메모리에서만 gradient-capture transformation으로 교체하여 gradient를 반환하고 parameter update는 0으로 만든다. 원래 checkpoint parameter 및 optimizer state의 전후 직렬화 SHA256 일치를 assert했다. 어떤 checkpoint도 덮어쓰지 않는다.

각 seed에서 초기/50k/중간/최신 저장 policy로 full Gaussian action rollout512개씩, 총2048transition 수집. 과거 replay buffer는 저장되지 않았으므로 이것은 재구성한 작은 진단 pool이다. 최신 checkpoint의 actor/critic을 고정하고 batch256을128번 bootstrap sampling하여 gradient global L2를 측정한다. 환경당4seed×128=512batch, 총1024batch. Critic norm은 twin-Q 전체를 합친 값. 원래 Adam으로 학습한 gradient 통계를 추정하는 이전 분석과 달리 이번 결과는 명시적으로 계산한 raw gradient norm이다.

Ant 최신 checkpoint는4seed 모두350k. Humanoid는 seed0/2/3=450k, seed1=350k. 단계와 분포가 다르고 초기 데이터 비중이 높아 원래 학습 replay의 clipping 비율을 정확히 재현하지 않는다. 기존 학습은 재개하지 않았다.

## 큰 gradient만 제한하는 초기 후보

|대상|후보 범위|보수적 진단 후보|진단 pool에서 clipping 비율|
|---|---:|---:|---:|
|Ant actor|6–10|6|0.39%|
|Ant critic|1500–2000|2000|0.20%|
|Humanoid actor|60–75|75|0.59%|
|Humanoid critic|30000–40000|40000|0.98%|

실제 reward 개선을 검증한 최적값이 아니라, 이 pool의 대부분 gradient를 유지하며 큰 값만 제한하는 초기 설정이다. 임계값 선택에는 다른 seed/학습시점/실제 replay에서의 clipping 빈도 확인이 추가로 필요하다.

Ant actor5는37.1%, critic1000은33.6%를 자른다. Humanoid actor50은11.3%, critic20000은38.5%를 자른다. 이전 EMA-RMS 기반 critic 범위보다 이번 raw norm에 기반한 후보를 상향했다. 이 차이는 이동평균과 순간 gradient의 차이뿐 아니라 진단 pool과 학습 replay 분포 차이도 포함한다.

이전 Humanoid seed1 Adam EMA-RMS777은 이번128batch의 raw gradient에서 재현되지 않았다(seed1 max62.7). 드문 큰 gradient 또는 다른 replay 상태의 영향일 수 있으나 원인을 확정하지 못했다.777을 일반적인 순간 gradient 크기로 취급하지 않는다.

현재 ac_grad_norm은 actor/critic 공통값이므로 위 분리 설정을 한 필드로 적용할 수 없다. 이번 작업에서는 clipping 설정을 적용하거나 학습을 다시 시작하지 않았다.

원시 수치: ant.json, humanoid.json. 통합 통계: summary.json. 진단 구현: probe.py.
