# MaxEntDP 공개 구현과 OptiQ AntMaze 비교

점검일: 2026-09-22. 학습 코드, 실험 설정, 큐 및 중지된 작업은 변경하지 않았다.
이 폴더에는 비교 보고서와 읽기용 소스 사본만 저장했다.

## 1. 비교 범위와 가장 중요한 제한

- MaxEntDP: 공식 저장소 `diffusionyes/MaxEntDP`, HEAD
  `8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077`.
- 원격 공개 branch는 main 하나이며, 공개 commit 4개를 모두 검색했다.
  논문 D.2/Figure 8의 online AntMaze 전용 환경, 등록 코드, reward wrapper,
  실행 명령 및 trajectory plotting 코드는 찾지 못했다.
- `examples/states/configs/iql_antmaze_config.py`는 `IQLLearner` 설정이다.
  `train_offline.py`의 `D4RLDataset` 및 reward-1 처리와 연결되는 offline 코드이며,
  MaxEntDP의 online 다중 경로 실험 설정이 아니다.
- 우리 현재 direct-gmm-trg: 서버180 HEAD
  `30f4db1cf9929974dacbdbca7c9c5abd5c1bd346`.
  최근 취소한 routefast 보조 보상/초기화 curriculum은 비교 대상으로 삼지 않았다.
- 실제 결과는 원래 dense-only 100k(`f47cedad3c757f37893fa6de0466882ebb9673e3`)와
  dense+NovelD 100k(`592cad778c20647030629c84dade4e54f79df67f`)를 구분했다.

따라서 아래에서 **논문에 명시된 AntMaze 사실**, **공개 공통 코드의 기본값**,
**우리 실제 실행 설정**을 구분한다. 공통 기본값을 Figure 8의 정확한 실행 설정으로
확정하지 않는다.

## 2. 논문에 명시된 AntMaze 조건

Appendix D.2 / Figure 8,9:

- DDiffPG의 AntMaze 환경 사용.
- Sparse reward를 가장 가까운 목표까지의 거리 페널티인 dense reward로 변경.
- 1M environment interactions 이후 MaxEntDP와 SAC의 궤적 및 state coverage 비교.
- AntMaze 전용 temperature, reward 스케일/오프셋, 성공 보너스 유무, 종료 조건,
  reset 분포, 관측 구성 및 Figure 8 평가 함수 호출은 공개 자료로 확정할 수 없다.

[논문 D.2](https://arxiv.org/html/2502.11612v3#A4.SS2)

우리 환경은 `antmaze/multimodal/env.py`:

- DDiffPG map/low-gear XML을 현대 MuJoCo로 이식.
- 29D 관측(qpos15+qvel14, xy 포함), 8D 행동[-1,1], 목표를 관측에 추가하지 않음.
- MuJoCo3.3.7/Gymnasium1.2.3, motor gear30, timestep.02, frame_skip5.
- reward = `-min_goal ||next_xy-goal||2`, scale1.
- 별도 success/healthy/locomotion reward와 control cost 없음.
- 성공 반경.5, 성공 즉시 종료. 넘어짐 자체는 종료하지 않음.
- horizon: v1=500, v3/v4=700. Timeout은 bootstrap.
- v1 초기 xy uniform[-2,2]^2, v3/v4 원점 고정; 기본 자세, 속도0.

이 조건들은 우리 코드에서 확인한 사실이다. MaxEntDP와 reward의 큰 형태는 같지만,
위 수치와 동역학까지 정확히 일치한다고 확인된 것은 아니다. 공개 의존성에는
Gym0.23.1, mujoco-py2.1.2.14 및 mujoco3.2.4가 함께 있으므로 AntMaze가 실제로
어느 backend로 실행되었는지도 해당 환경 소스 없이 단정하지 않는다.

## 3. 공개 MaxEntDP 학습 루프

`examples/states/train_score_matching_online.py`:

1. `gym.make(env_name)`으로 train/eval 환경 각 하나 생성(60,74행).
2. 행동을 [-1,1]로 rescale/clip. Replay capacity는 max_steps.
3. 기본 max_steps=1M, batch256, start_training=10k, utd_ratio=1.
4. `i < start_training`은 uniform random action, 이후 DDPM에서 행동 하나 샘플링.
5. 한 transition을 저장하고 `i >= start_training`부터 `agent.update(batch)` 한 번.
6. critic 2개를 각각 한 번, actor를 한 번 업데이트.
7. 매10k마다 기본10episode 평가. `eval_actions` 사용.

정확한 루프 경계 때문에 100k 종료 시 90,001회, 1M 종료 시 990,001회 update다.
이는 공통 기본값에 따른 계산이지 공개되지 않은 AntMaze run 로그의 측정값은 아니다.

주의: 이 스크립트의 `utd_ratio` flag는 `batch_size * utd_ratio`에만 사용된다.
MaxEntropyLearner.update는 이를 여러 minibatch로 나누어 반복하지 않는다.
따라서 이 flag를 올리면 default 이외에서는 이름과 달리 update 횟수가 늘지 않는다.

[공개 학습 루프](https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/train_score_matching_online.py#L92-L139)

## 4. 실제 학습 예산 차이

| 조건 | MaxEntDP 공통 루프를100k까지 | 우리 dense-only100k | 우리 NovelD100k |
|---|---:|---:|---:|
| 환경 수 | 1 | 1 | 256 |
| Batch | 256 | 256 | 4096 |
| Warmup | 10k | OptiQ/SAC/MEOW5k, MFPO10k | 8192 |
| Optimizer updates | 90,001 | 95,000 / MFPO90,000 | 전 방법2,872 |
| Collection 이후 update | 1transition당1회 | 1transition당1회 | 256transition당8회 |
| Update/transition | 약1 | 1 | 1/32 |

MaxEntDP 논문 AntMaze 그림은 **1M**, 우리 완료 파일은 **100k**다.
최근 NovelD 실험은 256환경에 총100k를 나눠 각 환경390~391step만 수집했다.
실제 네 방법의 progress.json 모두 training_episodes=0, training_successes=0이다.
v1 horizon500에 도달하기 전에 전체 예산이 소진된 구조다.

Batch4096은256의16배지만 update 수는 약31배 적다. 모델/target network/optimizer/
온도 조절 업데이트의 시간적 진행을 같은 것으로 볼 수 없다. 다만 기존 UTD1
실험도 모두0/100 성공이므로 **update 감소만으로 실패 원인이 확정되지는 않는다**.

근거: `antmaze/{results,noveld_results}/runs/v1-{optiq,sac,mfpo,meow}-s0/`
아래 config/result/progress.json; `antmaze/multimodal/ddiffpg_run.py:47`.

## 5. MaxEntDP와 현재 OptiQ TRG의 알고리즘 차이

| 항목 | MaxEntDP 공개 기본 구현 | 현재 OptiQ TRG |
|---|---|---|
| 정책 | DDPM, 20 denoising steps | random latent 조건부 box-truncated Gaussian, Direct marginal GMM NLL |
| Actor 주 네트워크 | 256×2 Mish + time Fourier64/encoder128×2 | 256×2 GELU, mean-head scale1 |
| Critic | scalar twin Q,256×2 ReLU | scalar twin Q,256×2 GELU |
| Actor 후보/target | QNE: 상태당500후보; replay action을 noise level별로 오염시킨 뒤 truncated Gaussian proposal | 상태당random latent64개, 현재 conditional mixture에서64후보 |
| Actor가 사용하는 Q | target twin Q의 min | current twin Q의 mean |
| Actor loss | Q-weighted noise target에 대한 MSE | softmax(Q/.25-logq) 가중 marginal NLL(beta1) |
| Critic entropy backup | true | `backup_mode=td`, entropy coefficient0 |
| 온도 | fixed .1가 공통 default; AntMaze전용값 미공개 | .25 |
| Actor LR | 시작3e-4,2Mupdate cosine decay | 고정3e-4 |
| Critic LR | 고정3e-4 | 고정3e-4 |
| gamma / target tau | .99 / .005 | .99 / .005 |
| Critic logπ 추정 | diffusion noise error로50samples×20time intervals | 현재 plain TD 경로에서 사용 안함 |
| 추가 탐색 | 공통 online 경로에 NovelD/RND/DACER 없음 | OptiQ DACER 행동잡음, 최근 profile은 전 방법NovelD도 추가 |

`T=20`은 MaxEntDP의 **diffusion step 수**이며, 우리 `temperature=.25`와
같은 종류의 설정이 아니다. MaxEntDP의 `temp=.1`도 AntMaze 실험값으로
확인되지 않았다. README의 Ant-v3=.05는 일반 locomotion Ant이고 AntMaze가 아니다.

### Critic objective 차이

MaxEntDP:

`y = r + gamma*mask*(min(Q1_target,Q2_target) - temp*logπ(a'|s'))`

현재 OptiQ TRG:

`y = r + gamma*mask*min(Q1_target,Q2_target)`

OptiQ의 actor는 여전히 temperature/density correction으로 분포를 학습한다.
따라서 'OptiQ에는 entropy 관련 메커니즘이 전혀 없다'는 설명은 틀리다.
정확한 차이는 **미래 상태의 정책 entropy를 critic return에 포함하지 않는다는 점**이다.
DACER의 현재 구현은 수집 행동에 잡음을 더하므로 이 Bellman objective를 바꾸지 않는다.
이 차이는 multi-step 경로 다양성을 해석할 때 검토할 가치가 있지만,
entropy backup을 켜면 현재 실패가 해결된다고 검증한 것은 아니다.

또한 OptiQ log sigma[-5,-1]은 conditional std 약[.0067,.368]을 뜻한다.
서로 다른 latent의 mean 차이도 정책 전체 분산에 기여하므로 이 std 범위만으로
정책 전체의 탐색 범위가 .368로 제한된다고 해석해서는 안 된다.

근거:

- [MaxEntDP 설정](https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/configs/max_entropy_learner_config.py)
- [MaxEntDP learner](https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/jaxrl5/agents/score_matching/max_entropy_learner.py): create65–177, critic179–240, actor242–336。
- 우리 현재 source의 `analysis_tools/experiments/20260921_gmm_trg_sweep/train.py:49`가 plain TD를 강제한다.
- `analysis_tools/experiments/20260920_truncated_mll/optiq_dime/algorithm.py`: critic328–415, actor599–739。
- `analysis_tools/experiments/20260921_gmm_trg_sweep/regulator.py:66`: DACER 행동 수집 잡음.

## 6. 평가 방식 차이

MaxEntDP 공통 기본 평가:

- `eval_action_selection=True`, `eval_candidate_num=10`.
- 매 행동마다 diffusion 후보10개 생성 → target twin-Q의 min이 최대인 후보 선택.
- 학습 수집에서는 후보 선택 없이 행동 하나를 샘플링한다.
- 공통 evaluate는 return 평균/표준편차만 반환한다. Figure8 경로 저장 코드는 없다.
- 논문 Table1 및 D.1도 M10을 기본으로 설명하지만, AntMaze Figure8이 같은 함수를
  호출했는지는 해당 script 부재로 직접 검증하지 못했다.

우리 `antmaze/multimodal/policy.py`:

- OptiQ: random latent + conditional sigma를 포함한 직접 정책 샘플;
  추가 DACER 행동잡음은 제거. mu-only는 별도 평가.
- SAC: stochastic Gaussian sampling.
- MFPO: native config에 eval_candidate_num10이 있어도 PolicyView가
  `eval_actions_sample_batch`를 직접 호출하므로 best-of10을 실제로 우회한다.
- MEOW: flow stochastic sampling.
- 최종 natural reset100episode 및 동일 full simulator state100episode를 별도 평가.
- 성공률, 목표/경로 비율, 궤적, 방문 coverage 저장.

이는 '정책 자체의 stochastic rollout 다양성'을 보기 위한 정의이지만,
Q선택 후의 MaxEntDP 공통 평가 정책과 동일하지 않다.
best-of10은 성공률과 경로 비율을 모두 바꿀 수 있으며 raw policy의 다양성과 구분해야 한다.

공개 jaxrl5 SACLearner의 기본 evaluate는 distribution mode를 사용하지만,
논문 C.1은 별도의 PyTorch SAC 저장소를 baseline 출처로 명시한다.
따라서 공개 jaxrl5 SAC 평가를 곧바로 Figure8의 SAC 평가 방식이라고 단정하지 않는다.
우리 SAC는 SB3 구현이므로 baseline 출처도 논문의 설명과 일치하지 않는다.

[MaxEntDP 평가 함수](https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/jaxrl5/agents/score_matching/max_entropy_learner.py#L338-L385)

## 7. 판단 및 확인 우선순위

1. 공개 자료만으로 MaxEntDP AntMaze의 정확한 환경·reward·temperature를 재현했다고
   주장할 수 없다. 우리 dense reward는 논문 설명에 맞춘 명시적인 구현이다.
2. 완료된100k pilot은 MaxEntDP 논문의1M 예산과 다르고, NovelD pilot은 update
   빈도까지 크게 달랐다. 이 상태의 실패를 알고리즘의 경로 다양성 한계로 해석하면 안 된다.
3. OptiQ와 MaxEntDP의 핵심 objective 차이는 plain TD/soft TD이며,
   평가 차이는 직접 정책샘플/best-of10이다. 단순히 batch나 temperature 수치만
   맞춘다고 이 차이들이 사라지지 않는다.
4. 비교를 설계한다면 환경 계약과 예산·update 빈도를 먼저 확정하고,
   stochastic rollout과 Q선택 rollout을 나눠 측정해야 한다.
   NovelD, entropy backup, 보상 변경은 각각 별도 조건으로 취급해야 효과를 분리할 수 있다.

이번 점검은 문서화만 수행했다. 어떤 학습, 재평가, 설정 변경 또는 재시작도 수행하지 않았다.
