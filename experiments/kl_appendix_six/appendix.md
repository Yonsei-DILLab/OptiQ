### Experimental details

**Targets.** We consider six fixed one-dimensional Boltzmann targets on $a\in[-10,10]$, with oracle $Q(a)=\alpha\log p^\star(a)$ and $\alpha=0.25$. No critic is learned, and target samples or mode labels are not provided during training. Targets were selected during an earlier screening experiment.

Four targets are Gaussian mixtures with component standard deviation $0.5$, normalized over the action domain:

| Target | Component means | Mixture weights |
|---|---|---|
| Three Gaussian modes | $(-4.25,0,4.25)$ | $(1/3,1/3,1/3)$ |
| Two offset Gaussian modes | $(0,4.25)$ | $(0.5,0.5)$ |
| Unequal Gaussian masses | $(-4.25,0,4.25)$ | $(0.2,0.5,0.3)$ |
| Minor Gaussian mode | $(-4.25,0,4.25)$ | $(0.1,0.65,0.25)$ |

For the two non-Gaussian targets, define $S(x)=(1+e^{-x})^{-1}$ and

$$
U(a;c,h)=\frac1h S\!\left(\frac{a-c}{h}\right)S\!\left(-\frac{a-c}{h}\right),
$$

$$
V(a;l,r,e_l,e_r,t)=S\!\left(\frac{a-l}{e_l}\right)
S\!\left(\frac{r-a}{e_r}\right)\exp\!\left[t\left(a-\frac{l+r}{2}\right)\right].
$$

Each shape is normalized individually on $[-10,10]$ before mixing:

$$
p^\star(a)=\sum_j w_j\frac{f_j(a)}{\int_{-10}^{10}f_j(x)\,dx}.
$$

The spike-and-ramp target uses $f_1(a)=U(a;-4.25,0.1)$, $f_2(a)=V(a;0.5,8,0.6,0.08,0.2)$, and weights $(0.2,0.8)$. The spike-plateau-ramp target uses $f_1(a)=U(a;-5,0.13)$, $f_2(a)=V(a;-1,1,0.12,0.12,0)$, $f_3(a)=V(a;3,7,0.25,0.05,0.4)$, and weights $(0.25,0.30,0.45)$.

**Actor and optimization.** Both methods use the same semi-implicit policy,

$$
q_\theta(a)=\mathbb E_{z\sim\mathcal N(0,1)}
[k_\theta(a\mid z)],
$$

where $k_\theta$ is a Gaussian conditioned on $[-10,10]$. A two-layer MLP with 256 units per layer and GELU activations outputs

$$
\mu_\theta(z)=10\tanh(h_\mu(z)),
\qquad
\log\sigma_\theta(z)=\operatorname{clip}(h_\sigma(z),-5,-1).
$$

The mean head uses a variance-scaling initializer with scale $3$, `fan_avg` mode, and a uniform distribution. The log-standard-deviation head is initialized to the constant $-1$.

Each update averages $B=32$ independent groups, each containing $N=128$ latent samples and $M=128$ action samples. We use Adam with learning rate $3\times10^{-4}$ for 100,000 updates and seeds $0,1,2,3$, with paired actor initialization across methods. The minor-Gaussian-mode figures use matched seeds 0–2 because Reverse seed 3 had not completed 100,000 updates; all other targets use four paired seeds. The unfinished checkpoint is not included in the final comparison.

**Reverse-KL objective and gradient.** Reverse KL minimizes

$$
\mathcal J_R(\theta)
=\mathrm{KL}(q_\theta\Vert p^\star)
=\mathbb E_{a\sim q_\theta}
\left[\log q_\theta(a)-\frac{Q(a)}{\alpha}\right]+C.
$$

For a reparameterized action $a_\theta=T_\theta(z,\epsilon)$, the gradient is

$$
\nabla_\theta\mathcal J_R
=\mathbb E_{z,\epsilon}
\left[
\left(s_\theta(a_\theta)-\frac1\alpha\nabla_aQ(a_\theta)\right)
\nabla_\theta a_\theta
\right],
\qquad
s_\theta(a)=\nabla_a\log q_\theta(a).
$$

This expression uses the score identity

$$
\mathbb E_{a\sim q_\theta}
\left[
\left.\nabla_\theta\log q_\theta(a)\right|_{a\ \mathrm{fixed}}
\right]=0.
$$

**Monte Carlo score estimation.** Since $q_\theta(a)$ is generally intractable, we draw $L$ independent latent Monte Carlo samples $\tilde z_\ell\sim\mathcal N(0,1)$, independently of the policy-action samples, and estimate

$$
\hat q_{\theta,L}(a)
=\frac1L\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell),
$$

$$
\hat s_{\theta,L}(a)
=\nabla_a\log\hat q_{\theta,L}(a)
=\frac{
\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell)
\nabla_a\log k_\theta(a\mid\tilde z_\ell)
}{\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell)}.
$$

For our truncated Gaussian kernels, at interior actions,

$$
\nabla_a\log k_\theta(a\mid\tilde z_\ell)
=\frac{\mu_\theta(\tilde z_\ell)-a}
{\sigma_\theta^2(\tilde z_\ell)}.
$$

The per-group gradient estimator is

$$
\hat g_R
=\frac1M\sum_{j=1}^{M}
\operatorname{sg}\!\left[
\hat s_{\theta,L}(a_{\theta,j})
-\frac1\alpha\nabla_aQ(a_{\theta,j})
\right]
\nabla_\theta a_{\theta,j},
$$

where $\operatorname{sg}$ denotes stop-gradient. It is implemented by differentiating the surrogate

$$
\mathcal L_R^{\mathrm{sur}}
=\frac1M\sum_{j=1}^{M}
a_{\theta,j}\,
\operatorname{sg}\!\left[
\hat s_{\theta,L}(a_{\theta,j})
-\frac1\alpha\nabla_aQ(a_{\theta,j})
\right].
$$

Gradients flow only through the reparameterized actions; the surrogate's numerical value is not the reverse-KL objective. We average the gradients over the $B$ groups and apply Adam.

Here, **$L$ is the number of latent MC samples used to estimate the marginal action score**, distinct from the $M$ policy actions used to estimate the objective gradient. The same $L$ latent samples are used to evaluate the scores of all $M$ actions within a group and are resampled independently for each group and update. We use $L=2^{20}$ while keeping $N=M=128$ and the actor parameterization fixed. Finite $L$ still yields an approximate gradient.

**Evaluation and score convergence.** Learned densities are visualized using 512-bin histograms of $2^{20}$ policy samples per seed, averaged across four seeds without smoothing. We report per-seed histogram TV and one-dimensional Wasserstein distance.

To assess score approximation, we freeze each final Reverse-KL actor and evaluate identical actions while varying the number of MC samples, $L=2^7,\ldots,2^{24}$. We perform 16 independent MC repetitions; within each repetition, increasing $L$ uses nested prefixes of the same latent sequence. The displayed actions are the seed-0 policy's 10th, 50th, and 90th percentiles. Curves show the MC mean, and shading shows the 10–90% range. Four additional independent estimates, each using $2^{24}$ latent MC samples, provide an empirical reference. Quantitative score errors are evaluated on 128 fixed policy actions for every seed. This reference is not an exact score, and convergence at the final checkpoints does not establish accuracy throughout training.
