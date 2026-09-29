# 이전 GMM40 성공 결과와 현재 TRG 실험 대조

이 문서는 기존 파일과 체크포인트를 읽은 진단입니다. 학습 코드·설정·큐는 변경하지 않았습니다.

## 확인된 결과

100k 업데이트, seed 0/1의 GT 3σ 내부 비율:

| 실험 | seed 0 | seed 1 |
|---|---:|---:|
| 이전 tanh-Gaussian Direct GMM, fixed64 | 97.06% | 93.00% |
| 이전 tanh-Gaussian Direct GMM, fresh64 | 76.20% | 79.34% |
| 현재 box-truncated Direct GMM/TRG, fresh64 | 54.75% | 71.64% |

이전 수치는 32,768개 평가 샘플, 현재 수치는 10,000개 평가 샘플에서 계산됐습니다. 타깃의 means/std/weights 배열은 정확히 동일합니다. 두 실험 모두 T=1, N=M=64, batch=256, actor 256×256 GELU, Adam 3e-4, 100k입니다. 과거 사용자가 첨부한 batch4 그림과 별개로, 로컬에서 완전한 설정/원자료를 찾은 이전 batch256 실험을 대조했습니다.

## 설정 차이

| 항목 | 이전 fixed/fresh 실험 | 현재 TRG |
|---|---|---|
| 조건부 분포 | 40*tanh(mu + sigma*epsilon) | [-1,1]에 절단된 Gaussian을 40배 |
| latent | fixed64 및 fresh64 각각 실험 | fresh64 |
| mean-head variance initializer scale | 1 | 1e-4 (Actor 기본값) |
| log sigma 범위 | [-5,1] | [-5,-1] |
| 초기 sigma | 0.5 | exp(-1), 상한과 동일 |
| teacher-only sigma floor | 0.05 | exp(-5)=0.006737947 |

분포와 좌표계가 다르므로 sigma의 숫자만 직접 비교해서는 안 됩니다. 현행 [-5,-1]은 사용자가 지정한 RL 기본값입니다. 현재 캠페인은 이 기본값을 유지한 TRG 비교이며 이전 성공 프로토콜의 재현이 아닙니다.

## 실제 체크포인트의 상한 포화

체크포인트/optimizer는 모두 실제 100,000 업데이트입니다. CPU에서 동일 actor를 재평가했으며 수동 Dense/GELU 전개와 원본 Flax Actor 출력 일치도 확인했습니다.

평가 latent 10,000개에서 raw log sigma > -1인 좌표 비율은 seed0 35.715%, seed1 12.94%입니다. 하나 이상의 좌표가 해당하는 latent 비율은 70.61%, 22.08%입니다. 실제 코드의 jnp.clip(x,-5,-1) 미분은 x=-2에서 1, x=-1에서 0.5, x=0 또는 50에서 0입니다. 상한 밖 출력은 clip을 통한 sigma 학습 신호가 직접 전달되지 않습니다. 공유 trunk의 다른 경로는 여전히 파라미터를 변화시킬 수 있으므로 영구 고정이라고 단정하지 않습니다.

현재 상한 sigma의 물리 좌표 scale parameter는 40*exp(-1)=14.715이며 타깃 Gaussian의 sigma는 1.313입니다. 전자는 절단 전 scale parameter이며 절단 후 실제 표준편차와 같지 않습니다.

CPU 재평가와 저장된 GPU mu의 수치 차이도 JSON에 기록했습니다. cap 여부와 저장된 샘플의 연관 분석은 그 차이가 포함된 참고치이며 정확한 GPU 재평가에 의한 인과 검증은 아닙니다. 핵심 진단은 실제 raw sigma 출력의 큰 상한 초과와 clip gradient 단절입니다.

latent 변경만으로는 이전 fresh 대비 추가 하락을 설명할 수 없습니다. 상한 포화가 유력한 문제지만 분포 변경, 초기화, teacher floor, RNG가 함께 달라 각 요인의 성능 기여를 분리하려면 통제 실험이 필요합니다.

## 근거

- 이전 결과: artifacts/gmm40_fixed_fresh/RESULTS_KO.md, PROTOCOL.md, run.py, results/*/{manifest,metrics}.json
- 이전 소스 커밋: 184bd7e26736a3af136f23b028f52168a34e80e0
- 현재 소스 커밋: 87d5d8ff210569ace7e59bd8a53ad02141b67f0a
- 현재 adapter: gmm40/optiq_trg.py
- 현재 actor: analysis_tools/experiments/20260920_truncated_mll/optiq_dime/policy.py
- 진단 상세: sigma_checkpoint_audit.json
