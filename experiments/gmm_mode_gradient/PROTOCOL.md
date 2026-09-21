# 3-mode toy: responsibility로 mode별 gradient 선택

## 질문과 비교

최근 `20260920_gmm_gradient_interference_all` 실험의 동일 stationary Q·배포 소스·actor 초기화를 사용한다.
이번에는 측정만 하지 않고 실제 optimizer gradient를 변경한다. TRG MuJoCo actor로 바꾸지 않는다.

- baseline: 원본 Direct GMM marginal NLL Adam update, 변경 없음.
- mode_only: 각 latent의 최대 responsibility mode에서 온 output gradient만 유지.
- mode_confidence: 위 gradient에 해당 mode의 정규화된 responsibility를 곱함 (사용자 제안).

8크기:16×16,64×64,128×128,256×256,1024×1024,2048×2048,64×4096,2048×4096.
각 seed0–3, random initialization부터20K updates: 총96runs. 기존 결과는 보존하고
baseline도 같은 login4 실행환경에서 새로 돌린다. 최대8GPU, GPU당1worker·2CPU.
시간·seed별 성능에 따라 조기 중단하지 않는다. 초기 확인 후 전체 완료를 기다리지 않아도 된다.

## 유지하는 수식과 설정

$$f(a)=\frac13\sum_{c\in\{-0.6,0,0.6\}}\mathcal N(a;c,0.1^2),\quad Q(a)=0.25\log f(a),\quad a\in[-1,1].$$

정답은 f를[-1,1]에서 정규화한 분포. Basin L/C/R 경계는 −0.3,0.3이다.
Actor는 기존 $a=\tanh(\mu_\theta(z)+\sigma_\theta(z)\epsilon)$, state0,1D normal latent,
256×2 GELU, 초기sigma0.5, log-sigma[-5,1], Adam3e-4, state batch1.
N개 latent는 매 update 재표집한다. 고정 N개의 학습 가능한 독립 component가 아니다.
Proposal은 원본 conditional mixture, teacher-only sigma floor0.05, M개 IID component sampling.
$w_j=\mathrm{softmax}(Q(b_j)/0.25-\log q_F(b_j))$. RNG·정밀도·temperature·clip·teacher는 동일.

## H의 정확한 정의

$$\gamma_{ij}=\frac{k_i(b_j)}{\sum_l k_l(b_j)},\quad J_{ij}=w_j\gamma_{ij},\quad
\alpha_i=\sum_jJ_{ij},\quad H_{mi}=\frac{\sum_{j:b_j\in\mathcal B_m}J_{ij}}{\alpha_i}.$$

$\sum_m H_{mi}=1$인 유효 component에서 $m_i=\arg\max_m H_{mi}$, $c_i=\max_mH_{mi}$.
alpha가 수치적으로0이면 confidence와 gradient0. 같은 최대값에서는 작은 mode index를 택하며
tie 빈도를 저장한다. H는 teacher에 의해 가중된 responsibility 비율이다. 가까운 거리나
conditional 자체의 basin 확률과 같지 않으며, 후보가 mode를 놓치면 confidence도 오해를 줄 수 있다.

## 정확히 어느 gradient를 선택하는가

$g_{i,m}$는 공유 parameter의 gradient가 아니라 output $(\mu_i,\log\sigma_i)$에 대한
mode m의 marginal NLL gradient다. 여기서 선택·가중한 뒤 network Jacobian으로 합친다.

$$\tilde g_i=c_i g_{i,m_i},\qquad
\tilde g_\theta=\sum_i\left(\frac{\partial(\mu_i,\log\sigma_i)}{\partial\theta}\right)^\top\tilde g_i.$$

실제 구현은 현재 theta에서 계산한 posterior·mode·confidence·teacher를 모두 stop-gradient하고,
다음 surrogate를 미분한다. mode_only에서는 $c_i=1$을 사용한다.

$$L_{\rm route}(\theta)=-\sum_{ij}\mathrm{stopgrad}\left[w_j\gamma_{ij}c_i\mathbf1\{b_j\in\mathcal B_{m_i}\}\right]\log k_{\theta,i}(b_j).$$

추가1/N, mode별 weight 재정규화, gradient norm 보정은 하지 않는다. Tanh Jacobian은
고정 target에서 actor와 무관하므로 원본 pre-tanh log-density로 같은 gradient를 얻는다.
이는 원래 marginal NLL의 gradient가 아니다. Common marginal NLL을 별도 기록하고
surrogate loss의 수치를 원래 loss와 우열 비교하지 않는다.

선택된 신호도 공유 network를 통과하므로 다른 담당 latent를 움직일 수 있다.
Adam의 이전 momentum도 유지한다. 따라서 '다른 mode가 움직이지 않는다'를 보장하지 않는다.
초기 거의 대칭인 actor에서 모든 latent가 같은 mode를 선택하는 실패 가능성도 그대로 기록한다.
수학적으로 알려진3개 basin을 이용하는 toy 개입이며, 일반 RL의 mode 발견을 해결한 것은 아니다.

## 진단과 그림

- 원본처럼32768개 실제 action·256-bin histogram, KDE 없음. 정답만 analytic density.
- 매1K + 상세시점에서 TV, basin mass, sigma, 고정2048latent의 representative mean/sigma/조건부 basin확률.
- 매100updates block 평균/마지막: common NLL, grad norm, confidence, 남긴 weight합, teacherESS/wmax, 할당mode별 latent비율.
- 상세시점0,1,10,100,500,1K,2K,5K,10K,15K,20K: H 원본·z정렬 heatmap,
  teacher basin mass, component별 output mode-gradient, 선택mode와 confidence, mode-gradient cosine.
- 동일 actor+Adam+teacher를 복사하여 세 방법의 한-update branch 비교: 고정latent의mean/sigma이동,
  원래 담당 mode의 질량 변화, 원본mode별NLL 변화, 동일noise histogram. Branch는 폐기.
- 기존 full/mode-only Adam 및 zero-gradient momentum control/SGD 대조도 보존.
- Confidence는 크기 효과와 분리: mode_only 대 mode_confidence가 동일gradient방향이 아니라
  component마다 다른 confidence로 reweight한 결과임을 명시.

## 검증·실행·저장

실행 전 commit/push heejoon; immutable source manifest에 full SHA 기록.
1) posterior mass conservation / H 정규화, mode-output Jacobian을 명시적으로 골라 VJP한 결과와
   surrogate gradient 일치; teacher·posterior·confidence stop-gradient 확인.
2) baseline의 원본step/block 일치; fixed-coefficient 유한차분; checkpoint+Adam+RNG 재개일치.
3) 세크기16×16,64×4096,2048×2048 CPU 수식 검증, 각 GPU worker의 세방법 실제 GPU 짧은 학습 확인.
검증이 통과한 뒤 독립 worker가 atomic task claim으로96runs를 나누어 실행한다.
사용 계정은 etc_qos 권한이 없어 bio파티션은 사용하지 않는다. CPU 수식 검증 후 big_qos GPU worker8개를 등록한다.
Validation dependency만 사용하고 개별 조건 간 completion gate는 없다.
Lustre 기존quota를 피해서 scratch2 사용; 기존MuJoCo/held jobs 건드리지 않음.
매500updates checkpoint, SIGTERM/SIGUSR1에서 최근block까지 저장. 재개는 같은 commit만 허용.
Run ID,SlurmID,노드,GPU,시각,source hash를 보관. 작은 toy이므로 별도 W&B run은 만들지 않음.
dildata에 immutable 소스와 완료체크포인트·histogram·진단·보고서를 저장한다.
MD 및 이미지 내장HTML 보고서는 동일 self-contained 형식으로 주기적으로 갱신한다.

## 저장 경로와 기존 작업 분리

기존 승인된 login4→dildata 읽기 전용 백업 연결을 재사용하기 위해, compute source/runtime은
`/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/mode_gradient_20260921`의
별도 새 폴더에 둔다. 상위 MuJoCo의 실행 코드·queue·설정은 수정하지 않는다.
Dildata의 독립 collector가 이 하위 폴더만 새 study 디렉터리로 회수하며 기존 collector의 종료와 무관하다.
첫fff685a는 etc_qos 제출 거절로 어떤 학습도 시작하지 않았다. 그 snapshot은 보존한다.
