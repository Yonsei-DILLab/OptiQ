# GMM40 Direct GMM: fixed versus fresh z64

요청 최종 설정: N=M64, T1/Qlogp, batch256, 100000 updates, seeds0/1 각각. actor256x256 GELU, mean-head variance init scale1, sigma init.5, teacher floor.05, zero latent skip, Adam3e-4. 업데이트당16384 teacher 후보를256 독립 묶음으로 정규화해 loss 평균.

|조건|seed|커버리지|3σ 내 비율|Mode-mass TV↓|MMD²↓|
|---|---:|---:|---:|---:|---:|
|fixed|0|40/40|97.06%|0.2257|0.01306|
|fixed|1|40/40|93.00%|0.1502|0.00494|
|fixed 평균|—|40/40|95.03%|0.1879|0.00900|
|fresh|0|40/40|76.20%|0.1019|0.00173|
|fresh|1|40/40|79.34%|0.0928|0.00138|
|fresh 평균|—|40/40|77.77%|0.0974|0.00156|

고정/비고정 모두 두 seed에서 최종40/40 target component를 커버. 고정은3σ내 집중도가 높고, 비고정은 mode-mass TV 및 multibandwidth MMD²가 낮다. 고정 seed0에는 target 밖의 작은 추가 cluster도 보이며, 커버리지100%가 완전한 분포 일치를 뜻하지 않는다. 각2seeds 결과로 일반적 우위를 단정하지 않는다.

Fixed는 정규분포에서1회 뽑은64 latent의 균등 prior로 평가. Fresh는 매step/batch마다 새64 Gaussian latent를 뽑고 평가도 새로운 Gaussian latent를 사용한다. 따라서 finite64 mixture와 continuous-latent policy의 실제 분포 비교이며, 같은 모델분포에서 gradient 분산만 바꾼 실험은 아니다.

Target은 v5-gmm40 a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71의 DiKL seed0 정의를 복사. x=40*tanh(u), 기준분포는(-40,40)^2로 조건화. 별도 NumPy수식과 JAX log-density 일치 확인. Branch의 original gmm40/evaluation.py assignments/MMD와 저장지표 일치 검증. 모든 체크포인트의 optimizer step과 run step100000,28개 sample snapshot finite, fixed bank 보존, paired initial mu/sigma/bank 동일 검증 완료.

평가32768샘플, 그림은필터링하지않은첫10000개. 파랑=어느 target center의3σ내, 주황=모든3σ밖. Coverage는nearest standardized center의3σ내 count≥max(10,.1*matched reference count); density local maxima40개의 엄밀한 개수는 아님. Mode-mass TV는3σ내 샘플에 조건화하므로 near fraction과 함께 해석. MMD²는2048샘플, RBF bandwidth1/2/5/10/20의 unbiased 평균.

batch32 예비실험도 모두100K완료되어 별도 보존. Final batch256은 fresh output으로 새로 수행. 첨부 과거그림(batch4)의 정확한 복제 실험이라고 주장하지 않으며, 이번에 명시한 batch256/init1조건 결과이다.

실행 전 서버 heejoon commit:184bd7e26736a3af136f23b028f52168a34e80e0. 원격 push 없음. 원격소스 /home/heechan/OptiQ-heejoon/experiments/gmm40_fixed_fresh. 원격결과 /home/heechan/optiq-experiments/gmm40-fixed-fresh-b256-20260920; 전체checkpoint/중간샘플 보존. 로컬 results/에는 final samples, manifest, metrics 저장.

[최종그림](comparison_100k.png) · [학습과정](training_progress.png) · [검증](verification.json)
