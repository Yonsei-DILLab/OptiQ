# DDiffPG intrinsic reward와 실행 설정 점검

점검일: 2026-09-21. 공식 저장소 HEAD 및 기존 환경 포트의 원본 SHA 모두
`7edd06c4799abbab0f8fa534c21deb56253b018e`.
실험 설정 변경이 아니라 논문·소스 대조 기록이다.

## 확인한 결론

DDiffPG는 RND 기반 NovelD intrinsic reward를 사용한다. 논문 §5.1은 RPG를
제외한 baseline에도 같은 intrinsic reward를 사용하고 RPG는 자체 RND를
사용한다고 설명한다. 공개 코드에서 DDiffPG, SAC, TD3, DIPO의 실제 critic
학습 경로에 연결된 것을 확인했다. 현재 우리 AntMaze 어댑터에는 이 모듈이
없다. 따라서 DDiffPG 결과와의 차이를 interaction 수만으로 설명할 수 없다.
다만 현재 환경은 MaxEntDP 문단을 참고한 dense reward 변형이므로, intrinsic
reward가 없다는 사실만으로 실패 원인 또는 구현 버그라고 단정할 수 없다.

논문: https://arxiv.org/html/2406.00681v1#S4.SS1
Baseline 조건: https://arxiv.org/html/2406.00681v1#S5.SS1

## 코드의 정확한 계산

`n(s) = ||predictor(phi(s)) - target(phi(s))||_2`

`r_int(s,s_next) = 0.01 * max(n(s_next) - 0.5*n(s), 0)`

- 기본 type=`noveld`, normalize=false, pos_enc=true, L=10.
- Ant의 x,y에 원좌표와 10개 주파수의 sin/cos를 붙이고 나머지 관측을 합친다.
  위치만으로 계산하는 RND가 아니라 전체 관측을 입력한다.
- predictor/고정 target은 512→256→128→128 MLP, hidden ELU.
- predictor는 MSE, AdamW 1e-4, gradient norm clip 1로 학습한다.
- replay minibatch의 현재/다음 상태로 novelty를 재계산하고, 두 상태를 합쳐
  predictor를 업데이트한다. 평가 return에 intrinsic reward를 넣지 않는다.
- SAC/TD3/DIPO: critic target의 reward에 환경 보상과 intrinsic reward를 더한다.
- DDiffPG: exploratory Q에는 intrinsic만, 다른 mode Q에는 환경+intrinsic을 사용한다.
  논문 §4.2의 탐색 전용 Q 설명과 실제 코드의 다른 mode reward를 구분했다.
- 이 구현에는 별도의 episode 방문 횟수 gate가 없다. 일반적인 NovelD 설명을
  추가로 가정하지 않는다.
- arXiv v1 HTML §4.1은 식 양쪽 novelty에 s'를 표기하지만 공개 코드는 위처럼
  다음 상태와 현재 상태를 비교한다. 구현 수식은 코드 기준으로 기재했다.

근거:

- [계산·업데이트](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/utils/intrinsic.py#L33-L94)
- [기본 설정](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/default.yaml#L31-L35)
- [RND 모델](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/models/mlp.py#L233-L267)
- [SAC 적용](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/sac.py#L106-L125)
- [DDiffPG Q별 적용](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/ddiffpg.py#L183-L235)

## 환경 보상과 현재 코드

논문 Appendix D는 sparse 0/1 보상을 설명하지만 공개 goal_reaching_env.py는
v1/v3/v4에서 미도달 0, 도달 10이다(v2 특정 목표는 20). reward_scale 기본은 1.
따라서 논문 설명의 0/1을 소스의 정확한 숫자로 인용하면 안 된다.

우리 `antmaze/multimodal/env.py:87`은 `-nearest_goal_distance`만 반환한다.
`antmaze/agents.py`의 SAC, MEOW, MFPO 및 OptiQ 어댑터에는 RND/NovelD가 없다.
OptiQ의 DACER entropy 기반 행동 탐색과 상태 novelty 기반 reward는 다른 장치다.

MaxEntDP Appendix D.2는 DDiffPG 미로에 dense 거리 penalty를 적용하고 1M
interaction 뒤 궤적을 비교한다고 명시한다. 이 문단만으로 RND 사용 여부를
확정할 수 없고, MFPO E.6의 상세 AntMaze reward도 공개된 내용 이상으로 추정하지 않는다.

- [원본 환경 보상](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/env/d4rl/locomotion/goal_reaching_env.py#L37-L60)
- [MaxEntDP D.2](https://arxiv.org/html/2502.11612v3#A4.SS2)

## 속도와 업데이트 예산

| 항목 | 우리 현재 설정 | 공개 원본 SAC 설정 |
|---|---:|---:|
| 학습 환경 수 | 1 | 256 |
| 한 수집 round의 transition 수 | 1 | 256 |
| batch size | 256 | 4096 |
| round당 optimizer update | 1 | 8 |
| transition당 optimizer update | 1 | 8/256=1/32 |
| transition당 sampled replay item 수 | 256 | 4096×8/256=128 |

원본 논문 Table 1의 UTD 8을 현재의 transition당 update 1과 직접 비교하면
틀린다. 공개 실행 루프는 horizon_len=1, num_envs=256이고, update_net 내부에서
update_times=8번 반복한다. 원본은 큰 batch와 훨씬 적은 optimizer 호출을
조합한다. 정확히 32배 빠르다는 뜻은 아니다: 각 update 비용, 물리 시뮬레이션,
평가, intrinsic 추가 비용, 병렬화 overhead가 다르다.

- [원본 SAC config](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/algo/sac_algo.yaml)
- [원본 수집·학습 루프](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/scripts/baselines_main.py#L130-L145)

추천하는 속도 검토 순서는 다음과 같다. 이는 미적용 제안이다.

1. 환경 병렬화 8~16개/run과 batch action inference를 먼저 benchmark한다.
   8 GPU에서 각 run에 256 CPU simulator를 무조건 붙이지 않는다.
2. 학습 알고리즘을 유지하려면 수집한 transition 수만큼 update를 유지하고
   여러 update를 JIT/compiled loop로 묶는다. batch만 키우면 update 수는 그대로여서
   총연산량이 늘 수 있다.
3. 연산 예산을 조절해도 된다면 16env/batch1024/round당4updates를 별도 후보로
   검토한다. replay item 수/transition은 현재와 같은 256이지만 optimizer
   update 수는 1/4이므로 원래 학습과 동등하지 않다. 같은 LR을 유지해도
   target-network update 빈도와 optimizer 동역학이 달라진다.
4. 평가 간격을 5k→25k로 줄이고 최종 trajectory 평가는 유지하는 방법은
   learner 설정을 바꾸지 않고 평가 비용을 줄인다.
5. RND 추가는 학습당 연산을 늘리며 sample efficiency 개선 가능성을 위한
   변경이다. 벽시계 속도 향상 기법으로 표현하면 안 된다.

정책 표현력 비교를 위해 intrinsic을 도입한다면 OptiQ/SAC/MFPO/MEOW 모두에
동일한 모듈·설정을 적용하고 각 run이 자기 데이터로 독립 학습해야 한다.
기존 dense-only 결과를 보존하고 dense+NovelD를 새 설정으로 명시해야 한다.
0.01은 원본 sparse reward의 계수이므로 dense 거리 보상에서도 적절하다고
보장할 수 없다. r_env/r_int의 실제 크기, coverage, 성공률을 확인해야 한다.
배치·intrinsic·병렬화 변경에 대한 새 실험은 이 점검 중 시작하지 않았다.
