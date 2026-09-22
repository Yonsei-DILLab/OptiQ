"""Small deterministic checks, not outcome-selecting experiment runs."""
import ast
from pathlib import Path
import numpy as np
import pytest


def test_legacy_actor_is_original_definition():
    root=Path(__file__).resolve().parents[2]
    def extract(p):
        src=p.read_text()
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('kernel_init','ImplicitActor')}
    assert extract(root/'optiq_dime/policy.py')==extract(Path(__file__).with_name('legacy_actor.py'))
    assert (root/'optiq_dime/transport.py').read_bytes()==Path(__file__).with_name('legacy_transport.py').read_bytes()


def test_trg_nll_gradient_matches_torch():
    import jax
    import jax.numpy as jnp
    import torch
    from .trg.distillation import direct_gmm_nll
    rng=np.random.default_rng(6)
    mu=rng.uniform(-.8,.8,(2,4,2)).astype('float32');ls=np.full_like(mu,-1.)
    a=rng.uniform(-.95,.95,(2,8,2)).astype('float32');w=rng.dirichlet(np.ones(8),2).astype('float32')
    value,grad=jax.value_and_grad(lambda m,l:direct_gmm_nll(m,l,jnp.array(a),jnp.array(w))[0],argnums=(0,1))(jnp.array(mu),jnp.array(ls))
    m=torch.tensor(mu,requires_grad=True);l=torch.tensor(ls,requires_grad=True);b=torch.tensor(a)
    standard=torch.distributions.Normal(0.,1.)
    z=standard.cdf((1-m)*torch.exp(-l))-standard.cdf((-1-m)*torch.exp(-l))
    ell=(-.5*((b[:,None]-m[:,:,None])*torch.exp(-l[:,:,None]))**2-l[:,:,None]-.5*np.log(2*np.pi)-torch.log(z[:,:,None])).sum(-1)
    loss=-(torch.tensor(w)*(torch.logsumexp(ell,dim=1)-np.log(4))).sum(-1).mean();loss.backward()
    np.testing.assert_allclose(float(value),loss.item(),rtol=3e-5)
    np.testing.assert_allclose(grad[0],m.grad.numpy(),rtol=2e-4,atol=2e-5)
    np.testing.assert_allclose(grad[1],l.grad.numpy(),rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('method',['optiq_trg','v5_gmm','optiq','legacy','monge','sql','mfpo'])
def test_checkpoint_and_eval_rng(method,tmp_path):
    import jax
    import jax.numpy as jnp
    from .adapters import make_agent
    class Oracle:
        def jax_log_prob(self,x):return -.5*jnp.sum((x/10)**2,axis=-1)
    agent=make_agent(dict(method=method,n=4,m=8,batch=1),Oracle(),0)
    agent.advance(1);checkpoint=tmp_path/'state.bin';agent.save(checkpoint)
    a=agent.evaluate_samples(16,123)[0];agent.advance(1)
    final=tmp_path/'final.bin';agent.save(final)
    reference=final.read_bytes()
    agent.restore(checkpoint)
    for seed in [1,2,3]:agent.evaluate_samples(16,seed)
    agent.advance(1);agent.save(final)
    assert reference==final.read_bytes(), 'Evaluation changed subsequent learner updates'
    agent.restore(checkpoint)
    np.testing.assert_array_equal(a,agent.evaluate_samples(16,123)[0])
