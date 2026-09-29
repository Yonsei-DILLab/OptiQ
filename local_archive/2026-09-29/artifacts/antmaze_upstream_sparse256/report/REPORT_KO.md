# AntMaze sparse256 결과 분석

확인 시각: 2026-09-23T11:35:49.176328+09:00
학습 source: `0ebd8d26c711d79723f457343b36eb8788bd87b8` · 모든 실험 seed0 한 개.

## 결론

**OptiQ/MFPO의 v1·v2 네 정책은 원래 예산까지 완료했으나 학습 중 도달0회, 최종 평가도 모두0%다. DIPO는 모든 미로에서 학습 중 실제 도달을 기록했고 v3 평가에서1/20 성공했지만, 네 run 모두 분포형 critic의 BCE target assertion으로 중단됐다.** 탐색 성과와 실행 오류를 분리해서 봐야 한다.

전체16개 중 완료4개, 실패4개, 미시작·보류8개다. 현재8개GPU 모두 학습 프로세스가 끝난 상태다. SAC4개, v3/v4 OptiQ/MFPO4개는 아직 시작하지 않았다. 첫DIPO실패가 빈슬롯 발생보다 빨라 controller의 shard 전체 pending 보류 규칙이 작동했다. 앞서 제시했던10~12시 완료 예상은 이 오류 이후 성립하지 않는다.

## 완료된 네 정책: 같은 예산 비교

정책마다 실제3,008,256 interactions, learner/RND93,752updates, batch4096,256env, sparse+NovelD0.01. 최종 표의 성공률은 **동일 full-state에서 직접 정책을100회 rollout**한 결과다. OptiQ는 random latent와 conditional sigma를 모두 포함한다. 모든 실패를 포함했다.

|미로|방법|interactions|최종 성공|학습 도달|학습 방문0.5m칸|학습 중 목표 최소거리|최종 최대이동거리 중앙값|
|---|---|---:|---:|---:|---:|---:|---:|
|v1|mfpo|3,008,256|0/100|0|257|2.48 m|3.79 m|
|v1|optiq|3,008,256|0/100|0|231|4.36 m|2.76 m|
|v2|mfpo|3,008,256|0/100|0|178|4.26 m|2.81 m|
|v2|optiq|3,008,256|0/100|0|121|2.52 m|1.60 m|

목표 최소거리는 장애물을 고려하지 않는 Euclidean 거리이며 성공 반경은0.5m다. 방문 칸 수는 한번이라도 방문한0.5m 격자 수로, 지속적인 방문이나 성공 경로 수를 뜻하지 않는다. 서로 다른 미로의 칸 수를 직접 성능 순위로 비교하면 안 된다.

- **v1:** OptiQ는 출발점 오른쪽 세로 통로에 대부분 머물렀다. MFPO는 특히 아래 우회 통로를 더 멀리 탐색했지만, 학습 중 가장 가까운 지점도 목표에서2.48m 떨어져 있었다. 최종 고정 시작 직접샘플100개에서 MFPO의 중앙장애물 윗길 진입10회/아랫길 진입2회, OptiQ는1회/0회였다. 이는 `x<-2 및 y<-2 / y>2`를 한번이라도 통과한 부분 진입이며 성공 경로가 아니다. 같은 rollout이 양쪽 조건을 모두 만족할 수도 있으므로 상호배타적 mode 비율로 해석하지 않는다.
- **v2:** OptiQ는 중앙에서 좁게 움직인다. MFPO는 왼쪽으로 더 멀리 나가며100회 중24회가x<-4를 통과했지만 목표까지 이어지지 않았다. OptiQ는좌/우4m 경계를 통과한 최종 rollout이0회다. 학습 탐색의 넓이는MFPO가178칸으로OptiQ121칸보다 약47% 넓다. 다만 OptiQ가 우측 목표에는 더 가까이 접근했으므로 넓이와 목표 진행도는 구분해야 한다.
- **성공 경로 한 개에 mode collapse했다고 결론 내릴 단계가 아니다.** 완료 정책에서는 성공 경로 자체가0개이고, 먼저 이동·탐색 범위가 제한되어 있다. 학습 중 탐색과 최종 정책 경로 다양성도 다르다.

![최종 직접 정책 궤적](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/final_policy_fixed_trajectories.png)

![학습 탐색영역](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/training_occupancy.png)

![누적 탐색영역](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/training_coverage_curves.png)

Native 평가(OptiQ mu-only, MFPO Q-best-of10), natural 초기 상태, OptiQ zero_z의 최종 성공률도 모두0%다. 따라서 직접샘플의 sigma 추가 하나만으로 설명할 결과는 아니다. [Native 그림](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/final_native_fixed_trajectories.png)

## DIPO: 부분 성과와 실패

|미로|오류 발생 interaction|학습 중 도달 최소횟수|마지막 저장 평가 step|직접 정책 평가 성공|
|---|---:|---:|---:|---:|
|v1|528,896|≥1|500,224|0/20|
|v3|1,099,520|≥3|1,000,192|1/20|
|v2|336,384|≥2|250,112|0/20|
|v4|915,712|≥3|750,080|0/20|

학습 중 도달은 오류 직전 마지막 progress 기록이므로 **하한**이다. 실패 직전4096step 구간의 추가 도달 여부는 복원할 수 없다. v3는 약1M평가에서 우상단 목표G2(12,-12)에1/20=5% 도달했다. v1은500k평가에서 목표에0.556m까지 접근했지만 성공반경0.5m에 못 미쳤다. DIPO는 네 미로에서 모두 훈련 중 적어도 한 번 목표를 발견했다.

DIPO native/direct-policy 두 평가에는 모두 원본 diffusion noise가 유지되고 같은 평가 RNG를 사용하므로 저장된 궤적이 완전히 같다. 따라서v3의1/20과1/20을 합쳐2/40의 독립평가로 계산하지 않았다. DIPO는중간20회, 완료방법은최종100회이고 예산도 달라 직접 최종성능 순위를 매길 수 없다. DIPO의 final checkpoint는 생성되지 않았다.

![DIPO 부분 궤적](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/dipo_partial_trajectories.png)

## 오류 진단

네DIPO 로그 모두 `/pytorch/aten/src/ATen/native/cuda/Loss.cu:91`의
`target_val >= zero && target_val <= one` assertion이다. 원본 `dipo.py`의 critic loss는 projected distribution을 BCE target으로 사용한다. 비동기CUDA때문에 Python stack은clip_grad 또는optimizer.step에 나타나지만 kernel 메시지는BCE target 확률의 범위 위반을 가리킨다.

변경하지 않은 원본 `distl_util.projection`을 CPU의 합성분포로 검사했다.51atom softmax분포, support[0,5], terminal reward10을 입력했을 때 projected probability 최대가 **1.0000003576**으로 나왔고4096개 분포 중1133개 값이1을 초과했다. BCE가CPU에서도 같은 범위 오류를 냈다. 이 예에서는NaN이0개였다. 부동소수점 합산오차로 이 오류가 생길 수 있음은 재현했다.

각 DIPO는 학습 도달 기록이 생긴 후 중단했고 원본 목표보상10/20은critic support상한5보다 크므로 성공 transition의 확률질량이끝 atom에 집중된다. **성공 샘플 이후 projection/BCE의 수치문제가 드러났을 가능성이 높다.** 실제 실패한미니배치는 저장되지 않아 정확한 trigger나오차크기는 확정하지 않는다. CUDA장애,학습률,gradient clipping문제로 단정할 근거는 없다. 임의clamp나설정변경/재실행은 이번분석에서 하지 않았다.

[재현 수치](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/projection_numerical_diagnostic.json)

## OptiQ 로그의 의미

완료OptiQ의sigma는exp(-1)=0.367879이고 최종상한포화비율은두환경모두100%다. latent에 따른mean분산은약1.76, 기록된latent mean variance fraction은약0.62라서 action이모두같은값으로붕괴했다고 보기는 어렵다. 하지만 action의무작위성은멀리이동하거나여러성공경로를배운다는뜻이아니다.

환경reward는학습전구간0이므로OptiQ가받은보상신호는NovelD뿐이었다. 마지막미니배치의reward_mean은v1약0.00386,v2약0.00368, T는0.25였다. 이정적인값만으로온도나NovelD의최적계수를정할수는없다. 현재결과는수집속도가빨라진것과정책이길을찾는것이별개임을보인다. [세부 로그](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/optiq_final_diagnostics.json)

## 다음 작업의 우선순위

1. DIPO의projection/BCE수치안정성을검증·수정하는것이먼저다. 목표도달은이미발생했으므로성공transition을정상학습할수있도록해야한다.
2. 보류중인SAC를실행해같은원본환경/수집설정의기준성능을확보해야한다. 현재SAC자료가없어4방법비교는불가능하다.
3. 이후OptiQ의작은행동랜덤성/넓은action분산과실제이동의관계를분석하고온도·탐색설정을대조한다. 이자료만으로NovelD계수를올리면해결된다고결론내리지않는다.

이문서는분석보고이며기존frozen학습소스나큐를변경하거나자동재시작하지않았다.

## 자료 검증 및 보존

서버원자료326개를다운로드해SHA256일치를검증했다.136개rolloutNPZ의success표시를실제xy목표반경·return·episode길이와대조했고fixed평가의초기full-state동일성을확인했다. 사전검사자료는결과에넣지않았다. 최종checkpoint4개는서버에보존되어있고remote저장후readback/SHA/환경보상검증proof를확인했다. 이로컬분석자료에는checkpoint본체를중복다운로드하지않았다.

199의완료OptiQ/MFPO두run은W&Bsyncsidecar로업로드완료된증거가있다. 실패DIPO의offline기록은서버에남아있으며완료run전용sidecar의업로드대상에들어가지않았다. 원격로그/frozen source를보존한다.

[전체 정량 JSON](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/analysis.json) · [CSV](/Users/yunheechan/Documents/ChatGPT/OptiQ/artifacts/antmaze_upstream_sparse256/report/completed_summary.csv)
