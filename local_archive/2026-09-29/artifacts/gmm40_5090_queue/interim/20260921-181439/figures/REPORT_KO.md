# GMM40 중간 결과



공통 업데이트 비교: 30,000, seed 0. 전체 24개 중 6개 완료, 4개 실행, 14개 대기인 스냅샷.

100k 결과는 완료된 seed만 표시하며 4시드 평균이 아닙니다. DIPO/MEOW 학습은 계속됩니다.

OptiQ: 현재 Direct GMM/TRG, N=M=64, random latent, log sigma [-5,-1].

평가: 각 패널에서 저장된 실제 full-policy 샘플 10,000개 전부 표시. 파랑은 임의의 GT 3σ 내부, 주황은 모두 외부.

Coverage는 3σ 내 샘플 수가 평가 코드의 임계치를 넘는 component 수입니다. 소수 샘플만 있는 모드도 포함할 수 있으므로 near와 MMD²를 함께 해석합니다.

비교는 동일 actor update 수 기준이며 baseline 구조와 Q 질의량/계산량은 서로 다릅니다.



![완료된 100k](completed_100k.png)

![동일 업데이트](equal_updates_seed0.png)

![학습곡선](learning_curves_interim.png)
