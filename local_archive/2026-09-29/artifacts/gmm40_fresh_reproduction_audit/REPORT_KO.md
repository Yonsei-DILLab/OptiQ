# GMM40: 과거 fresh 재현과 현재 TRG의 차이

2026-09-21 읽기 전용 코드/체크포인트 조사. 학습 코드와 실행 큐는 변경하지 않았다.

과거 검증 자료는 gmm40_fixed_fresh, frozen 184bd7e26736a3af136f23b028f52168a34e80e0.
당시 batch256 fixed seed0/1은 near97.06/93.00%, fresh seed0/1은76.20/79.34%, 모두coverage40/40.
사용자가 처음 첨부한 batch4 그림을 그대로 재현한 결과와는 구별한다.

현재 capm3 mean1 frozen17ae649의100k 결과는 seed0..3 coverage39/34/27/35,
near79.83/77.12/84.62/78.07%. 높은 near만으로 전체40모드 재현을 주장할 수 없다.
seed0 μ-only near89.27%,full79.83%. Gaussian noise를 없애도10.73%의 중심이GT3σ밖에 있어
전부가 sigma 탓은 아니다. 노이즈 추가 후 바깥 비율은20.17%다. 두 비율의 차이는 순변화이며
잡음 때문에 나간 샘플 수의 정확한 분해가 아니다. 잡음이 안으로 이동시키는 경우도 있다.

차이:
- 과거: x=40*tanh(mu(z)+sigma(z)*epsilon), sigma는pre-tanh좌표.
- 현재: tanh(mean_head)를 중심으로[-1,1]에 절단한Gaussian을40배, sigma는normalized action좌표.
- 과거 log sigma[-5,+1], 초기 sigma .5 (log≈-.693),teacher-only floor.05.
- 현재capm3 log sigma[-5,-3],초기-3,teacher floor exp(-5); 새capm1은상한/초기-1.
- 둘다mean-headscale1(현재대조군),256x2GELU,N=M64,batch256,T1,Adam3e-4,100k.
- 이전87d5d8f TRG는 mean-headscale1e-4; 새mean1과 혼동하지 않는다.
- seed 번호가 같아도 초기/학습 PRNG split 경로가 달라 정확히같은 초기값은 아니다.

현재sigma진단:
../gmm40_capm3_queue/diagnosis/sigma_checkpoint_audit.json
CPU10,000latent재평가에서rawlogσ>-3 좌표비율은seed0..3 100/91.73/83.78/81.36%.
GPU학습로그에서도seed0 sigma min=mean=max≈exp(-3).
CPU/GPUforward차이는JSON에 기록했으므로 비율은 CPU진단값이다.
jnp.clip의상한밖 직접미분은0. trunk의mu경로가raw출력을변화시킬수있으므로영구동결은아니다.
과거도hardclip사용했으나초기값이상한보다안쪽이었다. 현재상한초기화와클리핑의상호작용이
검증할우선문제이며모든성능차이를이원인하나로확정할수는없다.
GT물리σ1.31326,현재cap-3 Gaussian폭parameter1.99148. 절단전parameter이지실제SD는아니다.

fresh Gaussian latent에서 연속MLP는 서로다른모드로연결되는z영역사이에전이영역을만든다.
그영역에도latent확률질량이있으면mean이모드사이에나온다. Fixed bank평가는64개코드만뽑으므로
코드사이출력을직접평가하지않는다. 이것이fresh에서불가피하게20%오차가난다는뜻은아니다.
전이영역질량은작아질수있고,현재샘플만으로각bridging메커니즘의인과기여는확정하지못한다.
Direct marginalGMM NLL이모든모드의평균을MSE로회귀하는구현이라는설명은틀리다.

재현순서 제안(추가실험을실행한것아님):
1. 과거184bd7e의fresh프로파일그대로실행. 랜덤z학습/평가, mean1, tanhGaussian,
   logσ[-5,+1],초기σ.5,teacherfloor.05,256x2,batch256,N=M64,T1,Adam3e-4,100k.
   과거seed0/1 및 PRNG도유지. 목표는fresh76~79%/40모드 수준 재현이며fixed93~97%를성공기준으로혼용하지않는다.
2. 이후 TRG에서 초깃값을상한안쪽으로두는hardclip대조군과sigma부드러운범위매핑을비교.
   바운드매핑과초기화를한번에바꿔개별효과로주장하지않는다.
3. teacher탐색폭은policyσ와분리해대조. .05의물리폭은tanh좌표와box좌표에서같지않다.
   탐색폭을키운다고미발견모드회복이보장되지는않는다.

현재[-5,-1]+mean1 큐는이전tanhsquashed정책재현이아니라TRG mean초기화대조군이다.
