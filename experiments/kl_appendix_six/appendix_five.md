### Experimental details: five one-dimensional targets

**Target distributions.** We compare forward- and reverse-KL fitting on five fixed targets over $a\in[-10,10]$, ordered as rows 1–5 in the figures. The oracle value function is $Q(a)=\alpha\log p^\star(a)$ with $\alpha=0.25$, so its Boltzmann distribution is $p^\star$. No critic is learned, and training uses neither target samples nor mode labels. These illustrative targets were selected in a preliminary screening study.

For the Gaussian targets, we normalize $\sum_k w_k\mathcal N(a;c_k,0.5^2)$ over $[-10,10]$. For the non-Gaussian targets, define $S(x)=(1+e^{-x})^{-1}$ and

$$
U(a;c,h)=\frac1h S\!\left(\frac{a-c}{h}\right)S\!\left(-\frac{a-c}{h}\right),
$$

$$
V(a;l,r,e_l,e_r,t)=S\!\left(\frac{a-l}{e_l}\right)
S\!\left(\frac{r-a}{e_r}\right)
\exp\!\left[t\left(a-\frac{l+r}{2}\right)\right].
$$

Here, $U$ is a logistic spike; $V$ is a smooth plateau when $t=0$ and a rising ramp when $t>0$. Each non-Gaussian shape is normalized individually before mixing: $p^\star(a)=\sum_k w_k f_k(a)/\int_{-10}^{10}f_k(x)\,dx$.

| Figure row | Target | Centers or unnormalized shapes | Weights |
|---|---|---|---|
| 1 | Three Gaussian modes | $c=(-4.25,0,4.25)$ | $(1/3,1/3,1/3)$ |
| 2 | Spike + ramp | $U(a;-4.25,0.1)$; $V(a;0.5,8,0.6,0.08,0.2)$ | $(0.2,0.8)$ |
| 3 | Spike + plateau + ramp | $U(a;-5,0.13)$; $V(a;-1,1,0.12,0.12,0)$; $V(a;3,7,0.25,0.05,0.4)$ | $(0.25,0.30,0.45)$ |
| 4 | Two offset Gaussian modes | $c=(0,4.25)$ | $(0.5,0.5)$ |
| 5 | Unequal Gaussian masses | $c=(-4.25,0,4.25)$ | $(0.2,0.5,0.3)$ |

**Policy and optimization.** Both methods use the same semi-implicit policy $q_\theta(a)=\mathbb E_{z\sim\mathcal N(0,1)}[k_\theta(a\mid z)]$, where $k_\theta$ is a Gaussian truncated to $[-10,10]$. A two-hidden-layer MLP with 256 units per layer and GELU activations outputs

$$
\mu_\theta(z)=10\tanh(h_\mu(z)),\qquad
\log\sigma_\theta(z)=\operatorname{clip}(h_\sigma(z),-5,-1).
$$

The mean head uses a variance-scaling initializer with scale $3$, `fan_avg` mode, and a uniform distribution; the log-standard-deviation head initially outputs the constant $-1$. Each optimizer update averages $B=32$ independent groups with $N=M=128$. All five targets use 100,000 Adam updates, learning rate $3\times10^{-4}$, and seeds $0,1,2,3$, with paired actor initialization across methods.

**Reverse-KL optimization.** The objective and its pathwise gradient are

$$
\mathcal J_R(\theta)
=\mathrm{KL}(q_\theta\Vert p^\star)
=\mathbb E_{a\sim q_\theta}\left[\log q_\theta(a)-\frac{Q(a)}{\alpha}\right]+C,
$$

$$
\nabla_\theta\mathcal J_R
=\mathbb E_{z,\epsilon}\left[
\left(s_\theta(a_\theta)-\frac1\alpha\nabla_aQ(a_\theta)\right)
\nabla_\theta a_\theta\right],
\qquad s_\theta(a)=\nabla_a\log q_\theta(a),
$$

where $a_\theta=T_\theta(z,\epsilon)$ is a reparameterized policy action. The explicit parameter-score term has zero expectation under $q_\theta$.

We estimate the marginal action score using $L$ independent latent MC samples $\tilde z_\ell\sim\mathcal N(0,1)$, drawn independently of the policy-action samples:

$$
\hat s_{\theta,L}(a)
=\nabla_a\log\left[\frac1L\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell)\right]
=\frac{\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell)
\,\dfrac{\mu_\theta(\tilde z_\ell)-a}{\sigma_\theta^2(\tilde z_\ell)}}
{\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell)}.
$$

The Gaussian score expression applies at interior actions. We use $L=2^{20}$ throughout the final Reverse-KL runs, with $N=M=128$ unchanged. The same MC samples evaluate all $M$ action scores within a group and are resampled for each group and update. Thus, $L$ controls score approximation rather than the number of actions in the policy-gradient estimate. Finite $L$ still gives an approximate gradient.

Implementation differentiates the per-group surrogate

$$
\mathcal L_R^{\mathrm{sur}}
=\frac1M\sum_{j=1}^{M}a_{\theta,j}\,
\operatorname{sg}\!\left[
\hat s_{\theta,L}(a_{\theta,j})-\frac1\alpha\nabla_aQ(a_{\theta,j})\right],
$$

where $\operatorname{sg}$ denotes stop-gradient. Gradients flow only through the reparameterized actions; the $B$ group gradients are averaged before Adam. The surrogate's value is not the reverse-KL objective.

**Density and score evaluation.** Density panels average 512-bin histograms of $2^{20}$ fresh policy samples per seed across all four seeds, without KDE smoothing. Dashed black curves show the target densities. For score convergence, we freeze the final Reverse actors and vary $L=2^7,\ldots,2^{24}$ at identical actions. The plotted actions are the seed-0 policy's 10th, 50th, and 90th percentiles. Curves show the mean over 16 independent MC repetitions and shading their 10–90% range; within each repetition, increasing $L$ uses nested prefixes of the same latent sequence. Horizontal dotted lines average four additional independent estimates with $L=2^{24}$, and vertical dashed lines mark training $L=2^{20}$. This reference is empirical, and the convergence plots describe the final checkpoints rather than guaranteeing score accuracy throughout training.
