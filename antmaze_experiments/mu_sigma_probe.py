"""Frozen s0 latent/conditional variability, including truncated moments."""
import numpy as np


def diagnose(learner,output):
    import jax
    import jax.numpy as jnp
    from scipy.special import ndtr
    from .envs import make_one
    from optiq_dime.box_gaussian import sample_box
    env=make_one('v1',0,reward_profile=learner.reward_profile,random_init=False)
    obs=env.reset();env.close()
    p=learner.model.policy
    z=jax.random.normal(jax.random.PRNGKey(97000),(4096,8))
    mu,ls=p.actor_state.apply_fn({'params':p.actor_state.params},jnp.asarray(np.repeat(obs[None],4096,axis=0)),z)
    mu0,ls0=p.actor_state.apply_fn({'params':p.actor_state.params},jnp.asarray(obs[None]),jnp.zeros((1,8)))
    samples=np.asarray(sample_box(jax.random.PRNGKey(97001),jnp.broadcast_to(mu,(128,4096,8)),jnp.broadcast_to(ls,(128,4096,8))))
    mu=np.asarray(mu,dtype=np.float64);ls=np.asarray(ls,dtype=np.float64);sigma=np.exp(ls)
    lo=(-1-mu)/sigma;hi=(1-mu)/sigma
    phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    norm=ndtr(hi)-ndtr(lo);ratio=(phi(lo)-phi(hi))/norm
    conditional_mean=mu+sigma*ratio
    conditional_var=sigma**2*(1+(lo*phi(lo)-hi*phi(hi))/norm-ratio**2)
    between=conditional_mean.var(axis=0);within=conditional_var.mean(axis=0)
    np.savez_compressed(output/'mu_sigma.npz',obs=obs,z=np.asarray(z),mu=mu,sigma=sigma,log_std=ls,
                        mu_zero=np.asarray(mu0)[0],sigma_zero=np.exp(np.asarray(ls0)[0]),
                        conditional_mean=conditional_mean,conditional_var=conditional_var,
                        between=between,within=within,sample_actions=samples[0])
    return dict(latents=4096,conditional_draws_per_latent=128,initial_xy=obs[:2].tolist(),
        mu_mean=mu.mean(0).tolist(),mu_std=mu.std(0).tolist(),sigma_mean=sigma.mean(0).tolist(),
        sigma_min=sigma.min(0).tolist(),sigma_max=sigma.max(0).tolist(),
        log_std_lower_fraction=float(np.mean(ls<=-5+1e-5)),log_std_upper_fraction=float(np.mean(ls>=-1-1e-5)),
        between_variance=between.tolist(),within_variance=within.tolist(),
        latent_variance_share=float(between.sum()/(between+within).sum()),
        sample_total_var=samples.reshape(-1,8).var(0).tolist(),analytic_total_var=(between+within).tolist(),
        note='Exact truncated conditional moments; outer expectation approximated with4096 Gaussian latents. No policy update.')
