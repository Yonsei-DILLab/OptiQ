# 고정된 TRG actor에서 density-score의 L 수렴성

## 질문과 입력

학습을 더 하지 않고, 완료된 1D 3-mode 실험의20개 최종 checkpoint(forward4seeds, reverse training L128/256/1024/4096 각각4seeds)를 고정한다. 학습 소스 ae0c370f5d19765089030e62b64f4f0f60a30c37,20K updates만 입력으로 인정한다. 새 분석 코드는별도heejoon commit과별도immutable source폴더에 저장한다. 원래 학습 코드/결과를 변경하지 않는다.

이 분석의 score는 목표분포의 score가 아니라 **현재 학습된 policy의 action score**다.

$$\pi_\theta(a)=\mathbb E_{z\sim\mathcal N(0,1)}k_\theta(a\mid z),\qquad s_\theta(a)=\partial_a\log\pi_\theta(a).$$

N=M=128은 원래 학습 때의 값이다. 여기서는 optimizer update도, 새 teacher도 없으며 L만 바꿔 동일한 policy score를 추정한다. 잘 적합한 actor뿐 아니라 적합이 덜 된 actor도 모두 포함한다.

## 동일한 action과 독립 bank

각 checkpoint가 저장한 실제 float32 policy sample 중첫512개를 고정한다. 이것이학습 중 샘플링 영역에대한주평가다. 추가로모든checkpoint에공통인[-.999,.999]의257개격자를고정한다. 격자는policy가드물게방문하는영역도포함하는stress test다. 두평가를섞어하나의RMSE로보고하지않는다.

L=128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144를평가한다. 독립bank16개(NumPy PCG64 seeds92813–92828)를사용한다. 한bank내에서는작은L이큰L의prefix이므로action뿐아니라latent표본도paired비교다. 서로다른bank는독립적이며,수치적분reference에도MCbank를재사용하지않는다. 모든checkpoint에서동일한seed를써서checkpoint사이도common random numbers를사용한다.

$$\widehat s_L(a)=\frac{\sum_{l=1}^L k_\theta(a\mid z_l)(\mu_l-a)/\sigma_l^2}{\sum_{l=1}^L k_\theta(a\mid z_l)}.$$

각k는정확하게정규화된box-truncated Gaussian이다. Truncation normalizer는responsibility에는들어가지만action미분에서는상수다. log-sum-exp와128개단위chunk누적으로전체bank를동일하게계산한다. chunk별score를단순평균하지않는다. Actorforward는원래float32parameters/input으로수행하고kernel/sum은float64를써서MC오차와누적roundoff를구분한다. 같은최대bank의repeat0을float32로도계산해precision gap을보고한다.

## 큰 L을 정답이라고 가정하지 않는 reference

1D latent를직접적분한다.

$$s_{\rm ref}(a)=\frac{\int k_\theta(a\mid z)(\mu_\theta(z)-a)\sigma_\theta(z)^{-2}\varphi(z)dz}{\int k_\theta(a\mid z)\varphi(z)dz}.$$

[-12,12]에서composite16-pointGauss-Legendrequadrature를사용한다. panels256,512,1024,2048,4096,8192(총4096–131072nodes)로해상도를높인다. 모든고정action에서직전reference와의차이가0.001+0.0001|score|이하인refinement가연속두번나와야수치수렴했다고부른다. 그렇지않으면reference미확정으로표시하고권장L을결론내리지않는다.

잘린latent꼬리는P(|z|>12)로별도상한을계산한다. mu in[-1,1],sigma>=exp(-5),sigma<=exp(-1)이므로k<=2/(sigma_min sqrt(2pi))이고|partial_a k|<=kmax*2/sigma_min^2이다. 이상한저밀도action에서꼬리의상대기여가커질가능성도score오차상한으로검사한다. Reference가검증되었다는표현은이수치진단에한정되며해석적exact값이라는뜻은아니다.

## 수렴 판정(사전 지정)

512개의고정policy action에서S=sqrt(mean_a s_ref(a)^2)로두고,bank r의NRMSE는

$$e_r(L)=\frac{\sqrt{\operatorname{mean}_a(\widehat s_{L,r}(a)-s_{\rm ref}(a))^2}}{S}.$$

Score가0인개별지점에서상대오차를나누지않는다. 절대RMSE,99-percentileabsoluteerror,referenceRMS를함께보고한다. "16회중최대NRMSE"를주판정량으로사용한다. 이것은95%confidence bound가아니라반복실험의관찰된최댓값이다.

L->2Lscore변화도같은S로나눈다. 어떤L이통과하려면**그L과적어도한개의더큰L이평가되어야하고**,그L이상의모든측정값에서16banks의최대reference오차와최대doubling변화가둘다기준이하여야한다.

- 실용기준:5%.
- 엄격기준:1%.

262144까지엄격기준을만족하지못하면해당checkpoint만524288,1048576을추가계산한다. 동일seed의bank를이어사용한다. 최대범위에서도미통과면"충분한L을확인하지못함"으로표시한다. 격자기준1%의최소L도별도로보고한다.

Checkpoint별임계값과모든checkpoint를포괄하는최대임계값을제시한다. 일부checkpoint가미통과면그룹전체의1%수렴을주장하지않는다. 더큰L에서재차기준을벗어나면이전단계의통과판정을유지하지않는다.

## 그림과 해석

1. 모든checkpoint: L vs NRMSE(mean및min–max),1%/5%기준선;policy-probe와공통grid를구분.
2. 각학습방법의seed0: 같은action축에서score곡선,reference,bank간spread를표시.
3. 표: checkpoint별L4096오차,5%/1%최소L,quadrature안정성,float32roundoff.
4. 수렴했다고정확한Boltzmann분포학습까지보장되는것은아니다. 이분석은**이미학습된policy의score근사오차**만측정한다. 또한score자체오차가작아도s_ref-Q'/tau의잔여량이매우작으면reversegradient의상대오차는클수있어reference direction RMS도기록한다.
5. 최종checkpoint결과를다른차원,다른학습시점,다른Q에대한보편적인L보장으로일반화하지않는다.

## 검증, 실행, 보관

단일Gaussian의해석적score,weighted dense-vs-chunk계산,分할누적동일성,quadrature정규화,bankprefix동일성,threshold로직을검증한다. Actor/checkpoint SHA를전후비교해고정을확인한다. source/config/launcher/protocol을heejoon에commit·push한뒤login4 SLURM에서1GPU2CPU/job으로실행한다. node05와기존CUDA문제노드를제외하고CUDA환경을정리한다. 기존학습은중단하지않는다.

login4실험폴더: /scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/kl_score_convergence_20260923
중앙보관: dildata:/data1/heejoonorm/OptiQ/studies/20260923_kl_score_convergence/campaign
로컬보고서: reports/20260923_kl_score_convergence/report.md 및 figures.

## Reference 추가 해상도 검사

초기분석에서일부actor가131072quadraturenodes까지엄격한reference안정성조건을통과하지못했다. 판정tolerance는변경하지않는다. 해당actor에만quadraturenodes262144,524288,1048576,2097152,4194304,8388608을차례로추가한다. 원래MCbank/action/actor/score측정은전부그대로재사용한다. 정밀화reference로오차와최소L만다시계산하고runtime/refined_results에보관한다. 초기자료는runtime/results에그대로남긴다. 원래reference가통과한actor는재계산하지않는다. Reference2회연속안정성은동일하게유지하며추가최대범위에서도실패하면미확정으로남긴다. 후처리commit과원본MC파일SHA를별도REFINEMENT.json에기록한다.

## 一百万 bank 이후 제한된 추가 검사

Reference를 모두 확정한 뒤 엄격 1%를 통과하지 못한6개 actor에만 L=2,097,152,4,194,304,8,388,608을 추가한다. 사후 확장이라는 점을 명시하며 오차 판정기준을 완화하지 않는다. 원래16개 IID bank의 동일 prefix에서 누적합을 이어 계산한다. 첫128개 score를 재계산해 일치를 검증하고, 원래 모든 MC score 배열은 그대로 보존한다. 8,388,608에서도1%를 확인하지 못하면 그 사실을 결론으로 남긴다. 모든20개 actor에서 공통된 별도 latent65536개(seed761023)의 sigma 분포를 기록한다. Sigma와 추정 오차의 관계는 상관관계 진단이지 원인 확정 실험이 아니다. 결과는runtime/extended_results, 코드/commit은EXTENSION_SOURCE_MANIFEST.json에 보관한다.
