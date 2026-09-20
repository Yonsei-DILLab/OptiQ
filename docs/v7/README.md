# OptiQ v7 문서

[전체 알고리즘 설명](ALGORITHM_KO.md)이 현재 구현의 기준 문서입니다. 이론, 실제 loss와 gradient, 전체 RL 의사코드, 후보256개를 각각 독립 latent에서 생성하는 teacher, H4096·256→16·actor 16쌍, SAC soft TD, 설정·평가·저장·검증 범위를 함께 설명합니다.

- [공통 표기](NOTATION_KO.md): teacher $b_j$, importance weight $W_j$, OT $P_{ij}$와 행 조건부 $R_{ij}$, 추가 보정 $(1/H)/(\sum_jP_{ij})$.
- [수학 감사](MATHEMATICAL_AUDIT.md): 조건부 Boltzmann 목표, joint objective, 분포 복원의 충분조건과 source importance 보정.
- [조건부 loss 복원 기록](CONDITIONAL_RESTORE_KO.md): 사용자 요청에 따른 코드·설정·문서 복원, 소스 해시 및 중단한 비교 실험의 보존 기록.
- [구현 검증](VALIDATION_KO.md): 조건부 목적함수의 unit test와 Ant 짧은 검증 기록.
- [GMM 검증](GMM_VALIDATION.md): 고정 Q adapter 검증 기록.
- [의사코드](PSEUDOCODE.md): 알고리즘 흐름 요약.
- [조건부 loss의 GMM40 100K 결과](GMM100K_KO.md): T=alpha=1, epsilon0.1/0.01/0.001 결과. 현재 복원한 알고리즘의 이전 실행 기록.
- [중단한 marginal SAC 비교 기록](MARGINAL_SAC_VALIDATION_KO.md): 전체4096개 actor 밀도를 사용했던 비교 실험. 현재 기본 알고리즘이 아니다.
- [가져온 baseline 코드](../../gmm40-baseline/): 원본 코드·구조 유지, 실행하지 않음.
- [Baseline 복사 기록](BASELINE_COPY_MANIFEST.json): 원본 위치, 파일별 SHA-256 및 결과물 제외 목록.

현재 teacher는 256개의 fresh latent에서 Gaussian 행동을 하나씩 생성하고 전체 256-component proposal 밀도로 보정합니다. proposal Gaussian16개를 사용했던 중간 검증 결과는 과거 기록이며 최종 설정과 구분합니다. 본격 RL 성능 실험은 아직 실행하지 않았습니다.

현재 기본 potential은 상태별 ReLU 256×2 MLP이며 Adam 상태를 유지하여 actor 업데이트마다 한 번씩 학습합니다(dual LR 1e-4). Teacher는 매번 새로 만들고, actor와 dual은 같은 갱신 전 potential을 사용합니다. `ot_potential_mode=fresh_sinkhorn`은 이전 Sinkhorn100 대조군으로 남아 있습니다. Actor·critic LR, UTD와 표본 수는 유지합니다.

현재 actor는 선택된16개 latent의 조건부 Gaussian에 대해 `T log pi_i - Q - T log Pr(i|a,s)`를 source importance로 보정해 학습합니다. OT는 Boltzmann 목표를 latent별로 배정하며, Q와 배정함수 모두 새 행동에 대한 gradient를 제공합니다. 고정4096개 latent는 OT 적분점이며 전체4096개 actor 출력을 계산하지 않습니다. RL soft TD는 기존16-component self-inclusive mixture 밀도 추정을 사용합니다. 목표는 OT로 나눈 조건부 분포를 학습하여 Boltzmann policy를 모사하는 것이며, marginal SAC와 목적함수가 동일하다고 주장하지 않습니다.
