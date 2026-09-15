# OptiQ + best-k 실험 결과

2026-09-15: proposal-only seed0의 Ant/Hopper는1M을 완료했다. 두 환경의 seed1은
GPU0/1에서 시작했다. 행동 수집까지 결합한 조건은 별도로 검증 중이며,
현재 전체 결합의 성공이나 여러 시드에서의 개선을 확정하지 않는다.
새 seed1 두 실행도 warmup 이후 실제 로컬/온라인 설정 검증을 통과했다.

## Teacher proposal-only: seed0,1M 완료

원래 Gaussian 행동 수집(Kb1)을 유지하고 best-of-8 pilot으로 teacher proposal을
구성한다. 원래 Boltzmann 목표와 T, 실제 proposal에 대한 밀도보정, mean OT,
full conditional NLL, plain TD(Kt1), 기존 두 평가 모드를 유지한다.

900K–1M은 해당 구간21개 평가의 평균이다. AUC는5K부터1M까지 모든 평가를
사다리꼴 적분한 뒤 구간 길이로 나눈 평균이다. 단일 마지막 점수와 구분한다.

| 환경 | 평가 | meanOT 900K–1M | proposal8 900K–1M | meanOT AUC 평균 | proposal8 AUC 평균 |
|---|---|---:|---:|---:|---:|
| Ant | zero_z | 5193.3 | 4932.8 | 3493.0 | 2964.0 |
| Ant | stochastic_z | 5210.0 | 4902.4 | 3484.5 | 2969.2 |
| Hopper | zero_z | 3007.4 | 3552.1 | 2815.7 | 3109.5 |
| Hopper | stochastic_z | 3005.1 | 3530.3 | 2812.0 | 3099.9 |

Hopper는 마지막100K 평균과 전체 학습 평균이 높아 재현 확인을 진행한다.
Ant는 학습 속도와 마지막100K 평균 모두 meanOT보다 낮다. 이 결과를
두 환경 모두 개선한 방법으로 해석하지 않는다. 최종 한 번의 zero-z/stochastic-z
점수는 Ant4907.1/4905.4, Hopper3478.7/3646.4였다.

각 실행의 정상 종료,1M actor/critic checkpoint,995K 업데이트, 두 모드의
201×10 평가 결과, 소스/설정 일치 및 W&B finished 상태를 확인했다.
meanOT와 환경·정책 평가 시드 전체가 직접 일치한다. Hopper 원본 평가 파일은
기존 W&B experiment artifact에서 받아 확인했다. Hopper 기준 실행은 RTX3090,
이번 실행은 RTX4090이므로 GPU 수치 연산 차이가 있는 역사적 비교라는 한계는
남는다. 평가 에피소드와 시간 구간은 독립적인 학습 시드가 아니다.

- [Ant proposal-only seed0](https://wandb.ai/OptiQ/v5-bestk/runs/73vdpwn0), [meanOT seed0](https://wandb.ai/OptiQ/v5-jaehoon/runs/2kvdu0ch)
- [Hopper proposal-only seed0](https://wandb.ai/OptiQ/v5-bestk/runs/w25cmgy4), [meanOT seed0](https://wandb.ai/OptiQ/v5-jaehoon/runs/h7hn8alq)
- 실행 중 seed1: [Ant](https://wandb.ai/OptiQ/v5-bestk/runs/ytc5qo3v), [Hopper](https://wandb.ai/OptiQ/v5-bestk/runs/rsom5b3r)

## 행동 수집을 포함한 결합: 부분 결과

| 조건 | 검토 구간 | zero-z / stochastic-z 평균 | 같은 구간 meanOT | 판단 |
|---|---|---:|---:|---|
| Ant:250K부터 best8 수집 | 350–400K | 3527.6 /3354.0 | 4013.6 /4033.5 | 이전 구간보다 개선,500K까지 유지 |
| Hopper:250K부터 best8 수집 | 350–400K | 1817.7 /1876.8 | 3351.7 /3408.0 | 440K까지 회복 불안정,440300에서 중단 |
| Hopper:Gaussian/best8 수집50:50 | 100K 최근5회 | 629.9 /629.2 | 476.0 /464.6 | 초반 개선,150K 검토까지 유지 |

Ant 지연 수집은 같은350–400K proposal-only2727.9/2767.4보다 높지만,
meanOT 우월성은 입증하지 못했다. 저장된 actor/critic checkpoint와 전체 평가
결과는250K까지 proposal-only와 동일하다. 저장하지 않은 replay/RNG 상태를
직접 비교한 것은 아니다. [실행](https://wandb.ai/OptiQ/v5-bestk/runs/8xhn2lj7).

Hopper 혼합 수집은 best8을 적용할 확률이.5이며, 나머지는 원래 full Gaussian
행동이다. T=.01와 밀도보정 등 원래 학습 항은 모두 유지한다. 아직 초반이고
proposal-only보다 낮아 성공으로 확정하지 않는다.
[설계와 검증](BESTK_MIXED.md), [실행](https://wandb.ai/OptiQ/v5-bestk/runs/noo6mflw).

실패한 즉시 수집8 조건과 중단한 Hopper 지연 조건의 seed1은 취소했다.
기존 결과는 보존하며, 중단한 실행을1M 완료나 새 조건의 결과로 합치지 않는다.
현재 Ant 지연 및 Hopper 혼합 조건의 seed1은 각 seed0 뒤에 대기한다.
Random-pilot control은 아직 실행하지 않아 Q 선별 자체의 이득을 분리하지 못했다.
추가 시드 및 실제 행동 수집 결합의 지속적인 성능 검증이 남아 있다.
