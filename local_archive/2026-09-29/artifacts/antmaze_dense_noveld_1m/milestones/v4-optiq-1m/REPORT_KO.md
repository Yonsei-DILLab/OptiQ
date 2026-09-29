# AntMaze v4 OptiQ 1M 완료

training seed 0의 1M 학습과 995,000 learner/RND 업데이트가 완료됐다. 전체 checkpoint·replay·평가 archive를 로컬로 보관하고 검증했다.

같은 전체 초기 상태에서 각 100회 평가한 결과:

| 평가 | G1 성공 | G2 성공 | 실패 |
|---|---:|---:|---:|
| random z + conditional sigma | 99 | 0 | 1 |
| random z, mu-only | 94 | 0 | 6 |
| z=0, mu-only | 0 | 0 | 100 |

성공 경로는 모두 G1/upper-entry/upper-outer 하나였다. z=0은 같은 초기 상태에서 결정적으로 같은 궤적을 반복하며 목표에 가까이 갔지만 성공 반경에 들어가지 못했다. random-z 성공과 zero-z 실패를 혼동하지 않는다. 이는 단일 training seed의 결과이며 여러 목표를 사용하는 정책으로 볼 근거는 없다.

![평가 모드별 궤적](fixed-state-trajectories.png)

[그림 검증](fixed-state-verification.json). 학습 source는 `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`. 다른 진행 중 실험은 변경하지 않았다.
