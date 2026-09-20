# v7 구현·수식 검증 기록

현재 기본 경로: 고정 latent 적분점 4096, 새 proposal latent 256개에서
Gaussian을 하나씩 구성하고 각각 행동 1개 생성, 전체 256-component 밀도로
`softmax(Q/T-log q)` importance 재표집 16개, persistent potential로 4096×16 OT 배정,
teacher occurrence마다 latent를 선택하여 actor 16쌍 업데이트.
이론·구현·의사코드의 통합 문서는 [ALGORITHM_KO.md](ALGORITHM_KO.md)다.
기호는 [공통 표기](NOTATION_KO.md)를 따른다. 추가 student 표집 보정은
$(1/H)/(\sum_jP_{ij})$이며 teacher 가중치 $W_j$와 구분한다.

현재 RL critic은 사용자 정정에 따라 **soft TD**이며 고정 alpha=T이다.
생성 component를 포함한 실제 actor의 16-component mixture로 marginal
log-density를 추정한다. 환경 평가 return/zero-z·stochastic-z 프로토콜은 유지한다.
초기 plain-TD v7 검증 기록은 보존하되 최종 RL 설정의 검증으로 사용하지 않는다.

현재 persistent dual 복원 검증:

- 신규 semi-dual 테스트 9개와 기존 actor/transport 36개, 합계 45개 통과.
- $\sum_jP_{ij}-1/H$의 analytical gradient와 autodiff, 같은 pre-update potential의 actor/dual
  갱신, 2회 연속 Adam 상태, GMM 공유 상태 및 RL 상태별 potential 확인.
- H4096/M256/K16에서 teacher 행동·재표집 RNG를 유지하고 Sinkhorn 호출이
  없어졌는지 확인. Persistent 모드에 $(1/H)/(\sum_jP_{ij})\le K$ 상한이 없다는 반례도 검증.
- Persistent config/RL CPU 60개 통과. Dual state의 초기화·직렬화·필수 복원,
  기존 v5/fresh-Sinkhorn 경로 보존, alpha=T soft TD와 gradient 분리 확인.
- GPU1 실제 Ant 검증 2개 통과: persistent v7과 기존 v5 각각8 환경 step.
  v7은 actor/critic/dual 각6회 업데이트, dual Adam·checkpoint 정확복원 및
  dual 파일 누락 거부, soft TD와 paired 평가1/4/8을 확인했다.
  산출물: `/root/optiq-experiments/v7/validation/v7_persistent_dual_ant8_gpu1/`.
- GMM persistent CPU core/adapter·전체 checkpoint 재개 오차0, GPU batch256
  actor/dual100회 finite 검증 통과. Warm 약1.897ms/update이며 큰 importance
  분산의 관측값과 본 캠페인 경로는 [GMM100K_KO.md](GMM100K_KO.md)에 기록했다.

아래는 solver 복원 전 fresh-Sinkhorn 경로의 검증 기록이다:

- actor/transport 수학 테스트 36개 통과: teacher 밀도·Jacobian, 재표집,
  joint pair 선택, source importance 기대값·gradient, Q와 $\Pr(i\mid a,s)$의 action gradient,
  frozen teacher/OT, 고정 적분점, loss 항 분해, 수치적 Sinkhorn 성질.
- soft-TD 설정/RL unit 47개 통과: alpha=T, 실제 mixture log-density를 넣은
  Bellman target·critic loss, 온도 변화, target stop-gradient, teacher 개수와
  actor 학습 개수를 분리한 설정·metadata 연결.
- 최종 soft-TD Ant 통합 검증 1개 통과: 환경 8 step, actor/critic 각각 6회
  업데이트. H4096/M256/K16, proposal component256개 각각1회 표집 확인.
  검증용 batch2·warmup2·2-step episode.
  두 actor head 변화, Adam 상태/checkpoint 복원, paired 평가, timeout bootstrap 확인.
- 실제 마지막 Ant 로그: alpha=.25, entropy 추정 2.01194715,
  backup entropy 항 .50298679. `.25 × 2.01194715`와 일치한다.
- 이전 explicit v5의 실제 mean-OT/NLL·plain-TD Ant 회귀도 통과했다.

최종 RL 검증 산출물:
`/root/optiq-experiments/v7/validation/v7_p256_one_per_latent_ant8_gpu1/test_v7_actual_ant_full_quadra0/`

당시 GMM40은 공유 actor core를 직접 사용하는 고정-Q 검증이다. T=1, batch256,
200 update, actor·Adam/RNG 저장·재개 검증 완료. 실제 256개 component를
각각 한 번 사용했고, 행동·전체 mixture 밀도·checkpoint 재개가 일치했다.
최종 near 비율은 59.76%, coverage는 19/40, MMD²=.06520이었다.
동일한 초기 actor에서 시작한 이전 proposal Gaussian16개 방식의54.88%, 11/40,
.13787보다 개선됐지만, 초기 대비 coverage/MMD가 나빠져 전체 분포 복원
성공으로 볼 수 없다. 이 비교는 seed0의 짧은 구현 검증이다.
학습 시간은 compilation 포함 8.63초, warm update 약19.0ms였다.

GMM 산출물:
`/root/optiq-experiments/v7/gmm40-validation/oneperlatent256-h4096-k16-s0-200/`

이전 proposal Gaussian16개 방식의 추가 평가 전용 진단에서16개 teacher에 대한 source 질량 오차와 실제
Boltzmann 목표에 대한 source 질량 오차가 크게 달랐다. 상세 수식, 반례,
진단 수치는 [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md)에 정리했다.
해당 population 진단 수치를 현재 proposal Gaussian256개 결과로 인용하면 안 된다.

새 GMM100K 캠페인 및 solver별 속도는 [GMM100K_KO.md](GMM100K_KO.md)를 참조한다.
이 검증은 장기 MuJoCo 성능 실험이 아니다. 전체 RL 캠페인은 시작하지 않았으며
기존 실험/체크포인트는 보존했다. 현재 v7 변경은 커밋·푸시하지 않았다.
