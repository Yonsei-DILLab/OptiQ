import json
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from experiments.gmm_gradient_interference.core import initialize,heads
from optiq_dime.distillation import direct_gmm_nll
from optiq_dime.semi_implicit import ConditionalGaussianProposal
rng=np.random.default_rng(812);mu=rng.normal(size=(4,64,1)).astype('float32');ls=rng.uniform(-2,0,size=mu.shape).astype('float32');u=rng.normal(size=(4,64,1)).astype('float32');w=rng.uniform(size=(4,64)).astype('float32');w/=w.sum(1,keepdims=True)
resid=u[:,None]-mu[:,:,None];ell=(-.5*resid**2*np.exp(-2*ls[:,:,None])-ls[:,:,None]-.5*np.log(2*np.pi)).sum(-1)
logmix=logsumexp(ell,axis=1)-np.log(64);expected=-(w*logmix).sum(1).mean();post=np.exp(ell-logsumexp(ell,axis=1,keepdims=True));weighted=(post*w[:,None])[...,None]/4
expected_mu=(weighted*(-resid)*np.exp(-2*ls[:,:,None])).sum(2)
expected_ls=(weighted*(1-resid**2*np.exp(-2*ls[:,:,None]))).sum(2)
actual,grad=jax.value_and_grad(lambda m,l:direct_gmm_nll(m,l,jnp.array(u),jnp.array(w))[0],argnums=(0,1))(jnp.array(mu),jnp.array(ls))
np.testing.assert_allclose(actual,expected,atol=2e-6);np.testing.assert_allclose(grad[0],expected_mu,atol=2e-6);np.testing.assert_allclose(grad[1],expected_ls,atol=2e-6)
proposal=ConditionalGaussianProposal(jnp.array(mu),jnp.array(ls),.05);logq=np.asarray(proposal.log_prob(jnp.array(u)));logjac=(2*(np.log(2)-u-np.logaddexp(0,-2*u))).sum(-1)
np.testing.assert_allclose(logq,logmix-logjac,atol=3e-6)
print(json.dumps(dict(status='PASS',nll_error=float(abs(actual-expected)),mu_grad_max_error=float(abs(np.asarray(grad[0])-expected_mu).max()),ls_grad_max_error=float(abs(np.asarray(grad[1])-expected_ls).max()),proposal_jacobian_max_error=float(abs(logq-(logmix-logjac)).max()))))
