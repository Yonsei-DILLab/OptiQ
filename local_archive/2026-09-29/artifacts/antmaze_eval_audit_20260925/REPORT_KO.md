# AntMaze 경로 평가 방식 감사

v3/v4 frozen control source 23603a7, 학습 보상100×Euclidean progress, T=1, DACER/NovelD OFF, seed0, 약500k post-warmup. 학습/정책 가중치는 수정하지 않았다.

## 평가 시작 상태

원본 AntMaze reset_model은 random_init=false일 때 init_qpos/init_qvel을 그대로 사용한다. 학습 v3/v4와 기존 평가 모두 XY·자세·속도가 완전히 동일하다. 두 episode의 native-reset 사전확인에서 full state 고유값은1개였고, 원본 평가 및 추가200회 자료에서도 고유값1개다. 따라서 평가만 자세/속도를 지나치게 고정했다는 앞선 추정은 틀렸다.

## 실제 샘플링과 경로

Frozen JaxLearner.act는 매 행동 reset_noise()로 새 key를 만들고, policy mode에서 random latent와 conditional sigma를 모두 샘플링한다. zero_z·mu-only는 별도 모드다. 최종 원본 평가의 동일한 초기 관측100개에서 실제 첫 행동100개는 모두 고유했다.

|환경|이른 시점 직접정책 평가|추가 최종 직접정책 평가|반대 방향의 얕은 진입|첫 행동 std의 8D norm|
|---|---|---|---:|---:|
|v3 (left/right)|{'left': 14, 'right': 24, 'uncommitted': 2} / 40|{'right': 194, 'uncommitted': 6} / 200|0|1.027|
|v4 (lower/upper)|{'uncommitted': 7, 'lower': 19, 'upper': 14} / 40|{'upper': 197, 'uncommitted': 3} / 200|0|1.262|

v3의 얕은 진입은 x<−2, v4는 y<−2 기준이다. 각 환경200회에서 이마저0이었다. 기존 통로 판정 문턱(x<−8 등)만 지나치게 엄격해서 다른 쪽 경로가 누락된 것이 아니다.

같은 초기상태의 teacher probe에서도 latent별 conditional mean의 8차원 분산 합 제곱근은 v3 0.902, v4 1.405다. z가 완전히 무시된 것은 아니다. 다만 행동의 분산은 양쪽 경로를 끝까지 수행하는 확률과 다르며, 이 수치만으로 action 분포가 두 모드라고 말할 수 없다.

추가 CPU 평가와 원본 GPU 평가의 초기 한 스텝 XY 차이는 v3 최대1.71e−5, v4 최대6.01e−6. 장기 동역학의 수치 차이로 첫100 경로라벨 일치율은 각각92/100,98/100이었다. 두 실행 모두 반대 방향 진입0이라는 결론은 같다.

![초기/최종 경로 비교](route_eval_audit.png)

## 해석 범위

이 결과는 해당 체크포인트에서 초기상태를 고정한 direct 정책의 관측된 반대경로 비율이 낮다는 사실을 확인한다. 0/200은 정확한 확률0의 증명이 아니다. 버퍼에 남은 과거 전이와 과거 경로는 현재 actor π(a|s)의 두 경로 능력을 보장하지 않는다. 정책이 각 시점에 다양한 8D 행동을 내더라도, 복수 시점의 일관된 좌·우/상·하 제어가 경로를 결정한다. 어느 학습 동역학이 그 일관성을 없앴는지는 이 평가만으로 확정하지 않는다.

추가평가 원자료·해시와 첫 행동 자료는 data/, 재현 스크립트는 direct-gmm-trg-antmaze commit82a42ea에 있다.
