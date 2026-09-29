# MFPO AntMaze 100k 비교 조건 점검

점검일: 2026-09-21. 학습 코드·설정·기존 결과는 변경하지 않았다.

## 결론

현재 실패를 ‘AntMaze는 100k로 원래 불가능하다’고 해석할 근거는 없다.
현재 실행은 MFPO Figure 10의 재현이 아니라, DDiffPG에서 가져온 미로를
MuJoCo 3로 이식하고 거리 보상·DDiffPG NovelD 및 수집 스케줄·각 learner의
기본 설정을 결합한 별도 실험이다. 실패의 단일 원인은 아직 분리되지 않았다.

## 원문과 공개 코드에서 확인한 내용

- Figure 10의 100k 단위는 optimizer updates가 아니라 environment interactions다.
- 논문 Table 2의 공통 조건은 batch 256, UTD 1, hidden 256×3이다.
- 논문 E.7은 평가 시 MFPO 후보 행동 10개 중 Q에 따른 선택을 사용한다고 명시한다.
  AntMaze만 이 규칙에서 제외된다는 설명은 확인되지 않았다. Figure 10 실행 코드는
  공개 저장소에서 찾지 못했으므로 실제 AntMaze 평가 호출까지 검증한 것은 아니다.
- 공개 `train_online.py` 기본값은 단일 환경, warmup 10k, batch 256, UTD 1이다.
  이를 100k interactions까지 실행하면 약 90k learner updates다.
- 확인한 공식 MFPO 저장소 commit `d8b3977d29d4ef2d315e871337e5826f2eb79eb2`에는
  AntMaze 전용 환경·보상·실행 설정이 없다. 일반 Ant-v3 실행은 AntMaze 실행이 아니다.

근거: [MFPO 논문 Figure 10](https://arxiv.org/pdf/2604.14698#page=24),
[Table 2](https://arxiv.org/pdf/2604.14698#page=18),
[E.7](https://arxiv.org/pdf/2604.14698#page=22),
[공개 학습 루프](https://github.com/dongxiaoyi-xyz/MFPO/blob/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/train_online.py),
[공개 설정](https://github.com/dongxiaoyi-xyz/MFPO/blob/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/configs/mfpo_config.py).

## 실제 저장 결과

두 실험 모두 v1, training seed 0, 정확히 100k environment interactions다.
성공률은 각 정책에서 natural reset 100회 및 동일 full-state reset 100회를 따로 평가했다.

| 실행 | SAC updates | MFPO updates | SAC/MFPO 성공률 |
|---|---:|---:|---|
| 이전 dense-only | 95,000 | 90,000 | 두 reset 방식 모두 각각 0/100 |
| 최근 dense+NovelD | 2,872 | 2,872 | 두 reset 방식 모두 각각 0/100 |

이전 source: `f47cedad3c757f37893fa6de0466882ebb9673e3`.
최근 source: `592cad778c20647030629c84dade4e54f79df67f`.
근거 파일은 `antmaze/results/runs/v1-{sac,mfpo}-s0/result.json` 및
`antmaze/noveld_results/runs/v1-{sac,mfpo}-s0/{config,result,progress}.json`이다.

최근 실행은 256환경, batch4096, 수집 round당 8updates, warmup8192다.
따라서 UTD=8/256=1/32이고, 마지막 부분 round까지 포함해 359×8=2872updates다.
큰 batch는 optimizer·target-network·온도 업데이트 횟수를 보존하지 않는다.
약 90k updates 대비 약 31배 적다. 이 차이가 성능 차이의 전부임을 뜻하지 않는다.

100k를 256환경에 나눠 각 환경은 390~391 step만 진행했다. v1 horizon은 500이고
실제 모든 run의 `training_episodes=0`, `training_successes=0`이다.
총 transition 수는 맞지만 긴 학습 궤적 및 reset 경험의 구성은 달라졌다.
episode가 끝나지 않았다는 사실 자체가 off-policy 학습을 무효로 만드는 것은 아니다.

## 추가 불일치와 원인 후보

1. **환경 보상:** 현재는 `-nearest_goal_distance` + 학습 시 NovelD다.
   MFPO의 AntMaze 전용 보상과 동일하다고 검증되지 않았다. 중앙 장애물을 우회하려면
   일시적으로 직선거리가 늘어나는 구간이 생기므로 이 보상은 벽 근처 정체를 유도할
   가능성이 있다. 이는 기하에 따른 가설이며 실패 원인으로 확정한 것은 아니다.
2. **평가 정책:** 두 실행의 trajectory 평가는 MFPO에서 Q 후보 선택을 우회하고
   직접 stochastic sample을 사용한다. 한 정책의 경로 다양성 분석에는 의도가 있지만
   논문 기본 평가와 다른 조건이다. SAC도 실패했으므로 이것만으로 전부 설명할 수 없다.
3. **SAC 모델:** 현재는 SB3 SAC 256×2이며, 논문 공통 모델은 256×3이다.
   구현과 모델 크기의 차이가 존재하지만 단독 원인이라는 근거는 없다.
4. **NovelD 강도:** 계수 .01은 DDiffPG에서 가져왔지만 원본 sparse reward를
   dense 거리 보상으로 바꾼 후의 적정값은 검증되지 않았다. raw reward 평균의
   비율만으로 탐색 신호가 무효라고 단정할 수 없다.
5. **환경 이식:** 지도·gear30·frame-skip5·29D observation은 DDiffPG 소스와
   대조했다. MuJoCo 버전이 다르며, MFPO AntMaze와의 동역학 동등성은 미검증이다.
   XML의 custom init_qpos가 있다는 사실만으로 현재 reset 버그라고 할 수 없다.
   원본 Gym MujocoEnv 역시 simulator의 초기 qpos를 읽는다.

## 해석과 다음 확인 순서

최근 실험의 적은 update 수는 확정된 비교 조건 차이다. 그러나 이전 UTD1 실행도
실패했으므로 update 수 복원만으로 성공을 보장할 수 없다. 논문 AntMaze의 정확한
환경·보상 설정 확보와 원본 평가 방식 확인을 먼저 해야 한다. 기존 checkpoint를
MFPO best-of-10으로 추가 평가하면 학습 변경 없이 평가 방식 영향을 분리할 수 있다.
그 다음 batch256/UTD1/논문 모델 조건에서 성공 재현을 확인해야 정책 다양성의
알고리즘 비교가 가능하다. 이번 점검에서는 추가 학습이나 재평가를 실행하지 않았다.
