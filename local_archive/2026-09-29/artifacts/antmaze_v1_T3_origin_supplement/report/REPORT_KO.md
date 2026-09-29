# 공식 v1 동일 시작 상태에서의 성공 경로 유지

과거 T3, dense=-거리, DACER OFF 정책을 수정하지 않고 재평가했습니다. 원점에서 위치·자세·속도까지 모두 같은 상태로 시작하며, 최종 정책은 새로운 평가 난수로도 다시 확인합니다. v1의 원래 랜덤 시작 평가는 보존된 별도 주 평가입니다.

| 체크포인트 | 평가 | 위쪽 성공 | 아래쪽 성공 | 실패 |
|---|---|---:|---:|---:|
|2M checkpoint|policy|0|0|100|
|2M checkpoint|native|0|0|100|
|2.5M checkpoint|policy|1|1|98|
|2.5M checkpoint|native|10|19|71|
|2.75M checkpoint|policy|14|32|54|
|2.75M checkpoint|native|24|53|23|
|3.008M final|policy|20|71|9|
|3.008M final|native|22|72|6|
|3.008M final, new eval RNG|policy|26|66|8|
|3.008M final, new eval RNG|native|16|77|7|

![동일 상태 경로 유지](same_state_retention.png)

각 행은100회입니다. policy는 random z+conditional sigma이고 native는 random-z mu-only입니다. 외부 DACER 행동잡음, 학습 업데이트, best-of 선택은 없습니다. 전체 초기 상태 동일성, 보상 거리합, 실제 goal 반경, 원시 데이터 SHA256, 파라미터와 원본 체크포인트 보존을 검증했습니다.

이 결과의 범위는 공식 v1에서 하나의 학습 seed 정책이 같은 시작 상태에서도 두 성공 경로를 생성하는지입니다. 행동 밀도 자체의 다봉성, 모든 시작 상태, v3/v4, 여러 학습 seed의 재현성을 자동으로 입증하지 않습니다. 서로 다른 checkpoint나 평가 모드를 섞어 하나의 정책처럼 집계하지 않았습니다.

학습 source `5baa5b3463416cdfdcad465e8f862f3729802568`, 별도 평가 source `5c3c966f2bfdc83242b449534712c65fd4c7ac6e` 및2.75M 반경 검증 보정 `ae556841753c5baa1d58a227e8cc90fbf4aa129b`. actor/critic/TD/NLL/sampler 핵심6개 파일은 현재 소스와 바이트 단위로 같습니다. regulator 차이는 진단 기록 추가이며 해당 과거 정책에서는 DACER가 꺼져 있습니다.
