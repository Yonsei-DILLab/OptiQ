# v7 GMM40 구현 검증

[공통 표기](NOTATION_KO.md)의 teacher $b_j$, 가중치 $W_j$, OT $P_{ij}$,
행 조건부 $R_{ij}$를 사용한다.

`gmm40/v7.py`는 RL과 같은 `optiq_dime.persistent_transport.update_actor_persistent`를 호출하며,
actor 목적은 `optiq_dime.conditional_sac`를 공유한다.
Teacher, importance weighting, resampling, OT, actor 목적을 별도로 구현하지
않는다. 관측은 상수 0이며 Q callback은 `log p_GMM40(40*a)`다. 학습에는
Q 값/행동 미분만 사용하고 ground-truth 표본은 평가에만 사용한다.

Canonical 설정은 고정된 표준정규 latent 적분점 H=4096, 매번 새로 뽑는
독립적인 proposal latent 중심 256개에서 각각 한 번 생성한 teacher 후보 256개, W에 따른 OT 전
재표집 16개, OT 4096×16, actor joint pair 16개다. 각 압축 teacher 슬롯에서
$\Pr(i\mid\tilde b_j,s)=KP_{ij}$로 source i를 뽑는다. 이는 teacher 방향으로
정규화한 행 조건부 $R_{ij}$와 다르다. Teacher와 cost는 매번 새로 계산하고,
기본 `persistent_dual`은 상태에서 potential을 출력하는 ReLU 256×2 MLP와
Adam 상태를 유지하며 actor 업데이트당 semi-dual 한 번을 학습한다. GMM의
상태 입력은 공유 `ones[1,1]`, dual LR은 1e-4다. `fresh_sinkhorn`은 명시적
control로 유지한다.
고정 좌표는 `latent_seed=seed`로 재현하며 Sobol 좌표가 아니다.

현재 설정은 `proposal_components=256`, `proposals_per_component=1`,
`teacher_sampling_mode=stratified`다. 256개 component index를 각각 한 번씩 사용한다.
이는 이전 v6 `preot_m256_k16`의 `one_per_latent` 후보 생성 방식과 대응한다.
최초 v7의 proposal Gaussian16개 혼합물에서 iid 후보를 생성한200회 결과는 기존 디렉터리에 보존했으며,
현재 검증은 별도 `oneperlatent256` 디렉터리에서 새로 실행한다. 통합 설명은
[ALGORITHM_KO.md](ALGORITHM_KO.md)를 기준으로 한다.

Actor 목적은 다음과 같다. $a$는 선택된 actor 조건부에서 새로 생성한 행동이다.

$$
\operatorname{mean}\left[
\frac{1/H}{\sum_jP_{ij}}
\left(T\log\pi_i(a\mid s)-Q(s,a)-T\log\Pr(i\mid a,s)\right)\right].
$$

추가 student 표집 보정 $(1/H)/(\sum_jP_{ij})$는 detach한다. Self-normalize나
clipping은 없다. 이 보정은 teacher W를 두 번 적용하는 것이 아니며, 유한
반복 OT 분할 자체의 오차를 해결한다는 보장도 없다. NLL이나 sigma penalty는
추가하지 않는다. Q와 assignment는 새 actor 행동에 대해 미분한다.
row→column Sinkhorn control에서는 정확 산술상 $(1/H)/(\sum_jP_{ij})\le K$이지만, 현재 기본인
persistent dual에는 이 상한이 없다. 중요도 보정의 기대값 항등식은 유지되나
작은 경험적 source 질량에서 큰 중요도와 높은 분산이 생길 수 있어 실제
지표와 nonfinite 여부를 확인한다.

Actor는 기존 256×2, 초기 sigma=0.5, log sigma∈[-5,1], Adam 3e-4다.
GMM 온도 T=1, 기본 OT epsilon=0.1, 기본 B=256이다. epsilon0.01/0.001 비교는
명시적 override이며 온도와 표본 수를 유지한다. Persistent dual은 매번
Adam 한 번, explicit fresh-Sinkhorn control만 100회 반복한다.
작은 B를 명시하면 config에 smoke override로 기록한다. 업데이트 빈도나
학습률을 바꾸지 않는다. 고정 latent bank는 연속 prior의 유한 표본 근사다.

```bash
/root/.venv-optiq-mujoco/bin/python -m gmm40.validate_v7 --seed 0 --steps 200 \
  --potential-solver persistent_dual \
  --out /root/optiq-experiments/v7/gmm40-validation/persistent-dual-oneperlatent256-h4096-k16-s0-200
```

실행 권한을 받은 GPU를 supervisor로 관리하며 이 runner는 1–1000 업데이트만
허용한다. 기존 디렉터리는 덮어쓰지 않는다. 200회 결과는 구현 검증이며
40개 모드 복원 성공이나 RL 성능을 뜻하지 않는다.

검증 내용:

- CPU에서 shared update 직접 호출과 adapter의 actor/Adam/RNG 일치.
- 고정 source bank, 새 teacher, 4096×16 OT와 16 actor pair shape.
- Teacher RNG를 재구성하여 256개 component ID가 각각 한 번 사용되고,
  실제 teacher 행동과 전체 256-component mixture density가 일치하는지 확인.
- Q callback 값과 action gradient, full checkpoint의 1→2회 재개 일치.
- Checkpoint에 seed/settings 서명을 저장하고 다른 bank 설정으로 복원 거부.
- Persistent mode의 dual network·Adam·step도 함께 저장하고 actor와 같은
  update count를 확인한다. Solver/dual 설정이 다르면 복원을 거부한다.
- 매 평가 별도 RNG로 정책 및 mu-only 10,000개 표본 저장. 참조 seed는
  20260917, 평가 seed는 900000+학습 seed. 평가가 학습 RNG를 바꾸지 않음.
- Source TV와 row/column 오차, sampled log-importance 분산/최댓값/ESS,
  source 분포 아래 log-importance 분산과 importance 2차 모멘트의 로그 기록.

`validation.json`의 passed는 구현 smoke 통과만 의미한다. 초기/최종 near,
coverage, MMD² 및 모드 질량을 함께 봐야 한다. 중심 마커는 표본 위에 그린다.
Target JSON과 평가 지표는 기존 GMM40 fixture를 유지했고 이전 learner는
복사하거나 import하지 않는다.

체크포인트 재구성은 저장된 `config.json`으로 `make_agent`를 호출한 뒤
`restore(checkpoint_path)`를 사용한다. Actor·Adam·학습 RNG·update count와
설정/seed 서명을 함께 복원하며 source bank도 같은 seed로 재생성된다.

사용자가 별도로 요청한 긴 학습은 `python -m gmm40.train_v7`로 실행한다.
이 진입점은 같은 shared actor core와 canonical 설정을 사용하며, 짧은 검증
명령의 1–1000회 제한을 바꾸지 않는다. 기본 100K까지, 0/1K 및 매 5K에 전체
actor·dual·각 Adam·RNG checkpoint와 10,000개 평가 표본을 저장한다. `--resume-checkpoint`
는 과학적 설정과 training code hash를 검증한 후 새 출력 디렉터리에서 이어간다.
실행 중 `runtime.json`과 `training.jsonl`에 compile을 제외한 속도와 학습 지표를
기록한다. 자동 성능/속도 조기 종료는 없고 nonfinite 오류는 보존 후 실패로 기록한다.
