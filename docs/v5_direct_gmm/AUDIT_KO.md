# v5 Direct GMM 구현과 검증

## 출처와 변경 범위

- 기반 v5: `a5d5e281bcb10b5c31557073d95e03a1c5f2af4f`
- Direct GMM: heejoon `d7c53610b2cfb4508d4c3127598ee417ba1ff150`의
  `analysis_tools/studies/20260917_nonstationary_q/v5/optiq_dime/distillation.py`.
  `direct_gmm_nll` 함수는 그대로 가져왔다. heejoon 연구 폴더 전체의 다른
  실험 설정은 가져오지 않았다.
- GMM40 및 baseline adapter: v5-gmm40
  `a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71`의 `gmm40/`.
  외부 DiKL, DIPO, MEow, MFPO 저장소 URL/커밋은 `gmm40/baselines.json`에 고정한다.

`mujoco_v5_direct_gmm`는 `v5/final`을 상속하여 알고리즘 설정 중
`distillation_loss`만 바꾼다. 출력 경로와 W&B 이름은 실험 구분을 위해 별도다.
기존 `mujoco_v5`와 GMM40 `--method optiq`는 기존 OT 동작을 유지한다.
W&B 프로젝트 이름은 새 설정이며, 이 작업으로 온라인 학습 run을 만들지는 않았다.

## Teacher와 density correction

각 replay state에서 기존과 동일한 latent z_i ~ N(0,I), N=16개를 뽑고,
actor가 mu_i, sigma_i를 출력한다. Teacher만 sigma_F=max(sigma_i,0.05)를 사용한다.
M=64개 후보는 성분을 균등 IID 선택하여 u_j ~ N(mu_i,sigma_F,i²), a_j=tanh(u_j)로 생성한다.
실제로 샘플한 이 finite mixture의 action-space joint density는

    log q_F(a_j|s) = logsumexp_i log N(u_j; mu_i, diag(sigma_F,i²))
                    - log N - sum_d log(1 - tanh(u_jd)²)
    w_j = stop_gradient softmax_j(Q_mean(s,a_j)/T - log q_F(a_j|s))

이다. Gaussian log density는 action 차원을 먼저 합산하고 성분을 혼합한다.
성분별 평균, Jacobian 부호, 균등 혼합의 -log N, teacher floor 모두 포함된다.
원래 u를 보존하고 stable softplus Jacobian을 사용하므로 atanh(tanh(u)) 역변환을 하지 않는다.
샘플러와 log_prob가 같은 floor를 사용한다. beta=1이며 T=.25는 Q에만 적용된다.
Q, teacher parameters/candidates, 최종 w는 학생 loss의 역전파에서 차단한다.

이 보정은 proposal에서 뽑은 표본을 exp(Q/T) 목표에 맞추는 self-normalized
importance weighting이다. q_F는 뽑힌 N개 성분에 대한 정확한 밀도다.
연속 latent 전체를 적분한 실제 정책 marginal의 정확한 밀도라는 뜻은 아니다.
유한 M에서 self-normalization 편향과 낮은 ESS 문제는 남는다.

## OT를 제거한 학생 목적함수

기존 v5는 action-space 비용과 Sinkhorn으로 uniform student row marginal을 갖는
coupling P를 만들고, 정규화한 각 행 R_ij로 conditional Gaussian NLL을 학습한다.
Direct GMM은 cost/P/R/Sinkhorn/argmax projection을 계산하지 않고 다음을 최소화한다.

    L = - mean_b sum_j w_bj log[(1/N) sum_i N(u_bj; mu_bi,diag(sigma_bi²))]

Teacher는 동일하지만 학습 objective가 conditional NLL에서 marginal NLL로 바뀐다.
성분의 사후 책임도 r_ij=softmax_i(log N_i(u_j))가 logsumexp의 미분으로 생긴다.
책임도나 학생 mu/sigma를 detach하지 않는다. 두 head와 shared trunk가 함께 학습된다.
student loss는 teacher floor를 적용하지 않은 실제 actor sigma를 사용한다.

고정 teacher u에서 tanh Jacobian은 student parameter와 무관하므로 loss에서는
생략해도 gradient가 같다. **Teacher importance weight의 Jacobian은 생략할 수 없다.**
loss 수치 자체는 action-space NLL과 teacher-only 상수만큼 다르다.

Direct GMM에서는 OT의 성분별 사용량 균등 제약이 사라진다. 따라서 성분 중복이나
사용량 쏠림을 OT처럼 억제한다고 주장할 수 없다. `gmm_component_ess_fraction`,
`gmm_component_usage_min`, `gmm_underused_fraction`으로 이를 관측한다.
OT 전용 metric을 0으로 꾸며 기록하지 않는다.

## 유지된 v5 조건

Gaussian collection/TD action, student epsilon RNG draw, live twin-mean teacher Q,
target twin-min plain TD, gamma=.99, tau=.005, actor/critic 256×2 GELU,
Adam 3e-4, no gradient clipping, initial sigma=.5, beta=1, no anchors,
no uniform replacement, no annealing, no entropy backup/guard,
paired zero-z/sample-z mu-only evaluation을 유지한다.
기존 OT epsilon/iteration 필드는 설정 호환을 위해 남지만 Direct GMM 계산에는 쓰이지 않는다.

## GMM40와 baseline

`--method direct_gmm`과 `--method optiq`는 동일한 actor, proposal, Q와 RNG를 공유한다.
SAC/DIPO/MEow/MFPO adapter와 external baseline pin은 원본을 유지한다.
고정 Q는 DiKL 원래 40-component target의 log density, physical x=40*tanh(u),
teacher log density는 `log q_F(a)-2 log 40`이다. 이 scale 상수는 softmax에서 상쇄되지만
명시적으로 보존한다. T=1이 원래 target 복원 조건이며 T=.25는 p(x)^4를 목표로 바꾼다.
평가는 원래 bounded GMM40을 사용한다. Navigation은 learned Q이고 기본 T=.25다.
heejoon의 다른 scale/학습 budget은 이 adapter에 섞지 않았다.

```bash
# 각 서버에서; 기본은 기존 v5와 같은 JAX 0.4.33 / MuJoCo 환경
source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm
python scripts/verify_v5_direct_gmm.py benchmark=ant
python run_optiq_dime.py benchmark=ant seed=0

# 동일 seed/N/M/batch/T로 비교; 아래는 실제 학습 명령의 예시다.
python -m gmm40.run --method direct_gmm --name direct_gmm_s0 --seed 0
python -m gmm40.run --method optiq --name ot_v5_s0 --seed 0
# CUDA PyTorch baseline (SAC/DIPO/MEow)은 전용 환경 선택
source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm gmm40
# --method sac / dipo / meow / mfpo도 지원한다.
# 실제 장시간 실행은 supervisor 및 GPU lock wrapper를 사용한다.
```

별도 옵션 없이 MuJoCo는 1M 환경 step, fixed-Q GMM40은 100K update다.
장기 비교 실험은 이 세팅 작업에 포함하지 않는다.

## 검증 방법

`tests/test_direct_gmm.py`는 SciPy 기반 independent Gaussian mixture likelihood와
analytic responsibility gradient를 비교한다. teacher u/w 및 critic 역전파 차단,
Jacobian 생략 전후 학생 gradient 동일성, density floor/샘플러 일치,
v5 대비 teacher/RNG 동일성, Sinkhorn 호출 금지, OT parameter 무관성,
실제 Ant의 두 head 업데이트·plain TD·paired evaluation,
GMM40 체크포인트 다음 update 재현을 검증한다.
기존 v5·conditional proposal·distributional distillation regression도 함께 실행한다.
검증 결과와 서버별 commit은 별도 `VALIDATION_KO.md`에 기록한다.

수식과 구현의 일치 및 짧은 실행 검증은 장기 수렴, 성능 우위 또는 mode coverage의
보장이 아니다. 그러한 결론에는 여러 seed의 실제 비교 실험이 필요하다.

JAX pmap은 기존 v5의 mujoco 환경(NCCL 2.31.2)에서 두 서버 모두 통과했다.
CUDA PyTorch 전용 gmm40 환경(NCCL 2.26.2)은 단일 GPU baseline이 정상이나
pmap 통신 초기화 timeout이 재현되어 병렬 JAX 학습에는 사용하지 않는다.
두 환경의 JAX/Flax/Optax 버전은 동일하다. `activate.sh`는 Docker에서
`NCCL_CUMEM_HOST_ENABLE=0`을 기본 적용한다. 이 변수만 바꿔서는 구 환경의
timeout이 해결되지 않았으며, 특정 NCCL 버전만을 원인으로 확정하지 않는다.
[NVIDIA의 cuMem/컨테이너 설명](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html)을 참고했다.
