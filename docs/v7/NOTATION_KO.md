# OptiQ v7 공통 표기

현재 `ot_conditional_sac`의 기준 표기다. 기존 OptiQ의 teacher 행동 $b_j$,
importance weight $W_j$, OT coupling $P_{ij}$, 행별 조건부 $R_{ij}$를 유지한다.
과거 실험의 설정 키·로그 키·파일 이름은 재현성을 위해 유지한다.

| 표기 | 의미 |
|---|---|
| $s$, $a$ | 상태와 일반적인 정규화 행동 변수 |
| $z_i$, $H$ | standard-normal prior에서 뽑은 고정 OT 적분점과 수; H=4096 |
| $1/H$ | 적분점 하나의 prior 가중치; Gaussian 좌표 밀도 $p_0(z_i)$와 다름 |
| $\pi_i(a\mid s)$ | shared MLP의 $z_i$ 조건부 tanh-Gaussian; actor entropy에 사용 |
| $\pi_{\theta,H}(a\mid s)$ | $H^{-1}\sum_i\pi_i$, 이론상 finite-bank 정책 근사 |
| $\pi_\theta(a\mid s)$ | 연속 normal latent를 적분한 전체 실행 정책 |
| $\hat\pi_\theta(a\mid s)$ | soft TD의 fresh L=16-component self-inclusive density 근사 |
| $M$, $M_{\rm prop}$ | 원래 teacher 후보 수와 proposal Gaussian 수; 둘 다 256 |
| $b_j$, $v_j$ | 원래 teacher 행동과 저장된 pre-tanh 좌표; $b_j=\tanh v_j$ |
| $q(a\mid s)$ | teacher 후보의 전체 256-component proposal 행동 밀도 |
| $W_j$ | 원래 M개 후보의 normalized importance weight; 합 1 |
| $K$, $\tilde b_j$, $\tilde v_j$ | W로 재표집한 16개 occurrence와 좌표; 중복 유지 |
| $P_{ij}$ | H×K OT coupling; teacher 열 합 $1/K$ |
| $\sum_jP_{ij}$ | 현재 유한 teacher/map에서 source i의 평균 선택 확률 |
| $R_{ij}$ | $P_{ij}/\sum_kP_{ik}$; teacher 방향 행 조건부 |
| $f_i(s)$ | persistent state-conditioned dual MLP의 potential |
| $\Pr(i\mid a,s)$ | latent 방향 OT 배정 확률; $\sum_i\Pr(i\mid a,s)=1$ |
| $\pi_B(a\mid s)$ | Boltzmann 목표 $\exp(Q(s,a)/T)/Z(s)$ |
| $\bar P_i$ | 실제 목표의 population source 질량 $\int\pi_B(a\mid s)\Pr(i\mid a,s)da$ |
| $t_i(a\mid s)$ | 정규화 조건부 목표 $\pi_B(a\mid s)\Pr(i\mid a,s)/\bar P_i$ |
| $T=\alpha$, $\varepsilon_{\rm OT}$ | Boltzmann/entropy 온도와 OT regularization; 별개 값 |

GMM 물리 좌표는 $x=40a$다. Q와 density는 같은 좌표계를 사용하며
물리 density에는 tanh Jacobian과 $-d\log40$를 포함한다. 코드 `teacher_u`는 $v_j$다.

## Teacher와 OT coupling

$$
W_j=\frac{\exp(Q(s,b_j)/T-\log q(b_j\mid s))}
{\sum_{\ell=1}^M\exp(Q(s,b_\ell)/T-\log q(b_\ell\mid s))},\qquad
P_{ij}=\frac1K\Pr(i\mid\tilde b_j,s).
$$

W는 256→16 재표집 빈도에 반영한다. 이후 OT 열 질량은 1/K이며 W를 다시 곱하지 않는다.
P는 coupling 행렬에만 쓰고 proposal 개수 기호로 재사용하지 않는다.

$$
\Pr(i\mid a,s)=\operatorname{softmax}_i
\frac{f_i(s)-\|z_i-\operatorname{atanh}(a)\|^2}{\varepsilon_{\rm OT}},\qquad
R_{ij}=\frac{P_{ij}}{\sum_kP_{ik}}.
$$

R은 한 latent에서 teacher들을 보는 행 조건부, Pr은 한 행동에서 latent를 고르는
조건부다. 현재 actor loss의 새 행동 assignment는 **Pr이지 R이 아니다.**
Actor가 R-weighted teacher NLL로 회귀하는 것도 아니다.

## Source correction과 actor loss

$$
\text{source 보정계수}=\operatorname{sg}\left[\frac{1/H}{\sum_jP_{ij}}\right],
$$

$$
\mathcal L_{\rm actor}=\operatorname{mean}_{\rm selected\ pairs}
\left[\text{source 보정계수}\left(T\log\pi_i(a\mid s)-Q(s,a)
-T\log\Pr(i\mid a,s)\right)\right].
$$

Density는 선택된 actor Gaussian 하나의 actual σ와 Jacobian으로 계산한다.
Teacher q나 전체4096 mixture density가 아니다. 새 행동에 대한 Q와 Pr의 gradient는
통과시키고 potential·적분점·source weight는 고정한다.

Source correction은 teacher W의 반복 적용이 아니다. Frozen map에서 source 선택을
uniform quadrature prior로 보정하는 기대 gradient 항등식이다. Empirical
$\sum_jP_{ij}$와 population $\bar P_i$를 구분해야 하며, source correction만으로
$\bar P_i=1/H$ 또는 조건부 적합을 보장하지 않는다.

$\mathrm{Unif}_H=(1/H,\ldots,1/H)$와 $\bar P=(\bar P_1,\ldots,\bar P_H)$는
모집단 질량 오차의 TV/KL 계산에 사용한다. 로그 대응은
`source_mass` = $\sum_jP_{ij}$,
`source_importance` = $(1/H)/\sum_jP_{ij}$,
`teacher_log_w` = $\log W_j$다.
