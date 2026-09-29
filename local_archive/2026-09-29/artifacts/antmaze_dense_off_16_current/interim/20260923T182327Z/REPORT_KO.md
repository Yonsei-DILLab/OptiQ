# 현재 dense + NovelD OFF 실험 중간 평가

학습 source `a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3`. 단일 seed0. 기존 과거 모델의 재평가와 구분한다.
저장된 random-start 평가만 분석. 새 평가/학습 작업을 실행하지 않았다. 모든 실패 궤적을 포함했다.
OptiQ T=.01, actor/critic256x3, actorLR3e-4/criticLR5e-4, batch4096,256env,8updates/256transitions.
정책 직접 샘플은 conditional sigma 포함, mu-only는 별도 보조그림. 평가 외부 DACER 잡음 없음.

|작업|직접정책 평가 step|직접정책 성공|성공 경로|mu-only/native 평가 step|mu-only/native 성공|
|---|---:|---:|---|---:|---:|
|v1-dipo-s0|1,250,048|0/40|없음|1,250,048|0/40|
|v1-optiq-s0|3,008,256|98/100|G1/upper:98|3,008,256|99/100|
|v1-sac-s0|750,080|0/40|없음|750,080|0/40|
|v3-dipo-s0|1,000,192|0/40|없음|1,250,048|0/40|
|v3-optiq-s0|3,250,176|0/40|없음|3,250,176|0/40|
|v2-dipo-s0|500,224|34/40|goal(8,0)/central-corridor:34|500,224|34/40|
|v2-optiq-s0|2,250,240|39/40|goal(8,0)/central-corridor:39|2,250,240|38/40|
|v4-dipo-s0|500,224|0/40|없음|500,224|0/40|
|v4-optiq-s0|2,250,240|0/40|없음|2,250,240|0/40|

v1 최종은100회, 중간은40회이므로 표본 수가 다르다. 해당 정책·체크포인트의 관찰이며 드문 경로의 부재를 증명하지 않는다.
원시 궤적의 finite/padding, 실제 goal 반경 도달, dense 거리합 reward, goal ID, initial state random 여부, 전송SHA256를 검증했다.
학습진행률과 마지막 평가 step은 다를 수 있다. 경로 분류는 성공 episode의 미로 통로 횡단 위치에 따른다.
그림: optiq_latest_policy.png / optiq_latest_native.png / success_curves.png. 전체 체크포인트별 지표: analysis.json.
