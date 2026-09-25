import argparse,json
from pathlib import Path
from statistics import mean

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--analysis-commit',required=True);a=p.parse_args()
    d=json.loads((a.study/'ANALYSIS.json').read_text());fd=json.loads((a.out/'FIGURE_DATA.json').read_text())
    btv=mean(r['histogram_TV'] for r in fd['evaluation'] if r['kind']=='bad_structure');gtv=mean(r['histogram_TV'] for r in fd['evaluation'] if r['kind']=='good_structure')
    lines=[]
    for case in dict.fromkeys(r['case'] for r in d['rows']):
        bad=[r for r in d['rows'] if r['case']==case and r['kind']=='bad_structure'];good=[r for r in d['rows'] if r['case']==case and r['kind']=='good_structure'];r=bad[0]
        status='정답으로 탈출' if r['raw_kl']<1e-6 else ('GD endpoint에서 local minimum 확인' if max(x['raw_gradient_norm'] for x in bad)<1e-9 else '나쁜 영역에 잔류; 근처 정지점 추가 검증')
        lines.append(f"| {r['centers']} | {mean(x['raw_kl'] for x in bad):.6f} | {mean(x['raw_kl'] for x in good):.2e} | {status} |")
    body=r'''# K=3, mean-only Forward KL의 bad local minimum 재현

**재현에 성공했다.** Equal-weight 3-GMM에서 variance를 고정하고 mean 세 개만 학습해도, 초기화에 따라 정답과 나쁜 local minimum으로 갈라졌다. 대표 target은 중심 **(-1.5, 1.5, 6)**, 공통 표준편차 **0.5**다.

![초기화에 따른 최종 density 비교](mean_only_local_minimum.png)

검정 점선은 true target, 파란 선과 채움은 100K GD 후 학습된 분포다. 각 seed에서 실제 샘플 **2^20개**를 뽑아 **512 bins**로 histogram을 만들고 네 seed의 density를 평균했다. 학습 분포를 analytic density나 KDE로 그린 것이 아니다. Histogram 범위는 가장 바깥 target mean에서 양쪽으로 10 sigma까지이며, 그림의 x축만 5 sigma까지 확대했다.

## 1. 정확히 무엇을 학습했는가?

\[
p^*(x)=\frac13\sum_{c\in\{-1.5,1.5,6\}}\mathcal N(x;c,0.5^2),
\qquad
q_\mu(x)=\frac13\sum_{i=1}^{3}\mathcal N(x;\mu_i,0.5^2).
\]

\[
\mathcal L(\mu)=-\mathbb E_{x\sim p^*}[\log q_\mu(x)],
\qquad
\mu\leftarrow\mu-0.01\nabla_\mu\mathcal L(\mu).
\]

Target entropy는 상수이므로 이 NLL을 최소화하는 것은 Forward KL을 최소화하는 것과 같다. 정답은 세 mean을 target mean에 맞추는 것이며, 이때 KL=0이다.

| 항목 | 설정 |
|---|---|
| Trainable parameters | mean 3개만 |
| Mixture weight | 각각 1/3, 고정 |
| Standard deviation | 모두 0.5, 고정 |
| 분포의 정의역 | 실수 전체; truncated Gaussian 아님 |
| Optimizer | plain gradient descent; learning rate 0.01 |
| Updates | 100,000 |
| Seeds | 0, 1, 2, 3 |
| 나쁜 초기화 기본 위치 | (0, 5.8, 6.2) |
| 좋은 초기화 기본 위치 | (-1.5, 1.5, 6) |
| 초기화 jitter | 각 mean에 독립 Gaussian 표준편차 0.05; 조건 간 seed paired |
| 학습 expectation | float64, 4,097-point Simpson 적분 |
| 정밀 진단 | 16,385-point 적분 및 별도 adaptive quadrature |

기존 semi-implicit actor 실험과 달리 **SNIS, latent resampling, teacher proposal, neural network를 사용하지 않는다.** Population Forward KL 자체의 finite-mixture optimization 문제를 분리하기 위한 실험이다. Target sample의 우연한 누락으로 발생한 실패가 아니다.

## 2. 최종 결과

| 초기화 | 최종 mean (4 seeds 모두 같은 정지점) | Forward KL | Histogram TV 평균 |
|---|---|---:|---:|
| 나쁜 구조 | (-0.003496, 5.876889, 6.109359) | 2.769105 | BADTV |
| 정답 근처 | (-1.5, 1.5, 6) | 수치적으로 0 | GOODTV |

정답 근처 조건의 nonzero histogram TV는 샘플 histogram 평가의 오차다. Analytic CDF로 계산한 Wasserstein-1은 약 1.65e-14다. 나쁜 구조의 Wasserstein-1은 약 1.99425다.

나쁜 초기화에서는 첫 component가 왼쪽 두 target component 사이에 머물며 둘을 함께 설명하려 하고, 나머지 두 component는 오른쪽 target component 하나에 중복 배치된다. Variance가 고정되어 있으므로 왼쪽 두 봉우리를 하나의 넓은 Gaussian으로 잘 덮을 수도 없다.

![학습 중 KL 및 mean 위치](training_trajectory.png)

## 3. 단순히 학습이 느린 것이 아니라는 근거

대표 bad-start seed 0의 **원래 100K GD endpoint**에서:

\[
\|\nabla_\mu\mathcal L\|_2=6.27\times10^{-14},
\qquad
\lambda(\nabla^2_\mu\mathcal L)
\approx (0.0695863,\;0.619786,\;2.636850).
\]

Gradient는 사실상 0이고 Hessian은 positive definite다. 이 smooth population objective에서 **strict local minimum**을 지지하는 수치적 조건이다. 동시에 정답보다 KL이 2.7691 크므로 global optimum이 아니다.

- 독립적인 adaptive integration과 고정격자 계산의 최대 차이: 약 6.3e-16.
- 정지점 root polishing의 이동량: 약 9.0e-13. 학습 endpoint와 실질적으로 같다. 이 후처리는 학습 checkpoint를 바꾸지 않았다.
- Hessian 고유벡터 3개와 무작위 방향 24개, 양·음 방향 모두 검사했다.
- 반경 0.001 / 0.01 / 0.05에서 관측한 최소 loss 증가량은 각각 3.46e-8 / 3.31e-6 / 6.58e-5였다.

작은 gradient만 보고 local minimum으로 부른 것이 아니다. 이는 analytic theorem 증명 대신, 적분·미분·곡률을 교차검증한 수치적 재현이다.

## 4. 왜 이 구성인가?

기존 대칭 target (-4.25, 0, 4.25)에서는 특별히 나쁜 구조로 초기화해도 최종적으로 정답으로 돌아갔다. 그 결과는 local minimum의 증거로 쓸 수 없다.

이번에는 **두 target mode가 상대적으로 가깝고, 세 번째가 멀리 있는 비대칭 구조**를 사용했다. [Jin et al. (2016), §4.1.1](https://arxiv.org/pdf/1609.00978)의 3-GMM 구성에서 아이디어를 가져왔다. Fitted component 하나가 앞의 두 mode를 맡고 둘이 먼 mode를 맡는 배치가 나쁜 stationary basin을 만들 수 있다.

사용자가 언급한 [Local Minima Structures in Gaussian Mixture Models](https://arxiv.org/pdf/2009.13040)의 mean-only, equal-weight, known-common-variance 문제와 같은 종류의 목적함수다. 이번 수치는 해당 논문의 특정 수치 예제를 그대로 복제한 것이 아니라, 별도 명시한 비대칭 target으로 재현한 것이다.

**이 실험은 bad local minimum의 존재와 특정 초기화 주변에서의 재현성을 보인다. 무작위 초기화에서 얼마나 자주 실패하는지는 측정하지 않았다. 또한 semi-implicit actor가 이 target에서 성공하는지는 이번 실험으로 확인하지 않았다.**

## 5. 탐색한 모든 target

각 행마다 bad/good 초기화 각각 4 seeds, 총 9 x 2 x 4 = 72 trajectories를 완료했다. Sigma는 모두 0.5다. 모든 good-start는 정답을 복구했다.

| Target centers | Bad-start 평균 KL | Good-start 평균 KL | 해석 |
|---|---:|---:|---|
ALLROWS

D가 큰 일부 조건은 100K GD 시점의 gradient가 아직 약 1e-5이므로, 그 학습 endpoint 자체를 정확한 정지점이라고 부르지 않았다. 대신 근처 stationary point의 Hessian을 별도로 확인했다. 대표 figure는 네 seed 모두 수렴하고 양의 최소 고유값이 가장 큰 (-1.5, 1.5, 6) 조건이다. 두 번째 명확한 재현은 (-1, 1, 5), KL=1.140494, 최소 고유값 약 0.05549다.

## 6. 코드·결과 위치

- 학습 코드 commit: `d9b67bdeb5cb05e9763838fae3a0f817512cbf42` (`heejoon`, 실행 전 commit/push).
- 분석·그림 코드 commit: `ANALYSIS_COMMIT`.
- Validation Slurm job: `2330495`; production: `2330500_[0-8]`.
- login4: `/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/gmm3_mean_local_minima_20260925`.
- dildata: `/data1/heejoonorm/OptiQ/studies/20260925_gmm3_mean_local_minima`.
- Raw diagnostics: `ANALYSIS.json`; 개별 checkpoint/history: `campaign/runs/`.
- 그림은 PNG와 PDF로 함께 저장했다.
'''
    body=body.replace('BADTV',f'{btv:.5f}').replace('GOODTV',f'{gtv:.5f}').replace('ALLROWS','\n'.join(lines)).replace('ANALYSIS_COMMIT',a.analysis_commit)
    (a.out/'report.md').write_text(body)
    state=dict(complete_trajectories=72,planned_trajectories=72,steps=100000,training_commit='d9b67bdeb5cb05e9763838fae3a0f817512cbf42',analysis_commit=a.analysis_commit,job='2330500',selected_case=fd['case'],result='strict bad local minimum reproduced in all four bad-start seeds; good starts reach truth')
    (a.study/'CURRENT_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    (a.study/'EXECUTION.md').write_text(f"# Execution\n\nTraining commit {state['training_commit']}; analysis/figure commit {a.analysis_commit}.\nValidation 2330495 passed; production 2330500_[0-8] completed all 72 trajectories at 100K updates.\nNo running numerical source was modified. Postprocessing ran locally from the later analysis commit.\nCentral storage: dildata:/data1/heejoonorm/OptiQ/studies/20260925_gmm3_mean_local_minima.\n")
    print(a.out/'report.md')
if __name__=='__main__':main()
