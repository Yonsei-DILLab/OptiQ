import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr
from gmm40.target import Target
from gmm40.learner import GMM40Learner, Proposal
from gmm40.metrics import metrics
from gmm40.run import evaluation_steps
from ibolt.box_gaussian import mixture_log_prob


def test_paper_target_and_density():
    target=Target()
    values={k:target.metadata[k] for k in ['means','std','weights']}
    assert hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest() == \
        'a22a67f6f37d782fd75961e75f5df13c424fda9624245c99efe93862c929e87c'
    mass=np.prod(ndtr((40-target.means)/target.std[:,None])-
                 ndtr((-40-target.means)/target.std[:,None]),axis=1).mean()
    np.testing.assert_allclose(mass,np.exp(target.log_z),atol=1e-7)
    x=target.sample(256,13)
    assert x.shape==(256,2) and (np.abs(x)<40).all()
    np.testing.assert_array_equal(x,target.sample(256,13))
    np.testing.assert_allclose(target.jax_log_prob(jnp.asarray(x)),target.log_prob(x),atol=1e-5)


@pytest.mark.parametrize('floor', [.05, None])
def test_gmm40_update_and_checkpoint(tmp_path, floor):
    agent=GMM40Learner(Target(),seed=0,n=16,m=16,batch=2,teacher_std_floor=floor)
    assert sum(p.size for p in jax.tree_util.tree_leaves(agent.state.params))==133636
    mu,ls=agent.actor.apply({'params':agent.state.params},jnp.zeros((16,1)),jnp.ones((16,2)))
    np.testing.assert_allclose(ls,-4.)
    proposal=Proposal(mu[None],ls[None],.05)
    np.testing.assert_allclose(jnp.exp(proposal.effective_log_std()),.05,rtol=1e-6)
    info=agent.advance(2)
    assert agent.updates==int(agent.state.step)==2 and info['Q_evaluations']==64
    assert all(np.isfinite(v) for v in info.values())
    assert np.exp(-5)-1e-7 <= info['sigma_min'] <= info['sigma_max'] <= np.exp(-3.5)+1e-7
    key=np.asarray(agent.key).copy()
    full,_,extra=agent.evaluate_samples(256,900000)
    assert np.isfinite(full).all() and (np.abs(full)<=40).all()
    assert not np.array_equal(full,extra['mu_only'])
    np.testing.assert_array_equal(key,agent.key)
    path=tmp_path/'checkpoint.bin'
    agent.save(path)
    restored=GMM40Learner(Target(),seed=0,n=16,m=16,batch=2,teacher_std_floor=floor)
    restored.restore(path)
    agent.advance(1);restored.advance(1)
    for a,b in zip(jax.tree_util.tree_leaves(agent.state),jax.tree_util.tree_leaves(restored.state)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(agent.key,restored.key)


def test_proposal_without_floor_uses_actor_scales_for_sampling_and_density():
    mu=jnp.array([[[.1,-.2],[.3,.4]]])
    ls=jnp.array([[[-5.,-4.],[-3.5,-5.]]])
    proposal=Proposal(mu,ls,None)
    np.testing.assert_array_equal(proposal.effective_log_std(),ls)
    key=jax.random.PRNGKey(17)
    actions,_,_=proposal.sample(key,3,'exact')
    reference,_,_=Proposal(mu,ls,1e-12).sample(key,3,'exact')
    np.testing.assert_array_equal(actions,reference)
    np.testing.assert_array_equal(proposal.log_prob(actions),mixture_log_prob(actions,mu,ls))
    gradient=jax.grad(lambda scales: Proposal(mu,scales,None).effective_log_std().sum())(ls)
    np.testing.assert_array_equal(gradient,jnp.ones_like(ls))


def test_wider_actor_scale_bounds_without_proposal_floor():
    agent=GMM40Learner(Target(),n=16,m=16,batch=2,teacher_std_floor=None,log_std_max=-3.)
    from flax.core import unfreeze
    params=unfreeze(agent.state.params)
    params['log_std']['kernel']=jnp.zeros_like(params['log_std']['kernel'])
    for bias,expected in [([-3.25,0.],[-3.25,-3.]),([-6.,-3.25],[-5.,-3.25])]:
        params['log_std']['bias']=jnp.asarray(bias)
        _,ls=agent.actor.apply({'params':params},jnp.zeros((1,1)),jnp.zeros((1,2)))
        np.testing.assert_array_equal(ls,jnp.asarray([expected]))
        np.testing.assert_array_equal(Proposal(jnp.zeros_like(ls),ls,None).effective_log_std(),ls)


def test_gmm40_evaluation_protocol():
    assert evaluation_steps(100000)==[0,100,500,1000,2500,5000]+list(range(10000,100001,10000))
    target=Target()
    reference=target.sample(256,20260917)
    result=metrics(reference,target,reference,target.sample(256,20260917,bounded=False))
    assert result['sliced_wasserstein2']==0 and 0<=result['mode_coverage']<=40
    assert np.isfinite(result['mmd2']) and result['outside_fraction']==0


@pytest.mark.parametrize('initializer,gain', [('orthogonal',1.),('orthogonal_sqrt2',2.**.5)])
def test_hidden_orthogonal_initialization_preserves_heads_and_scale(initializer,gain):
    kwargs=dict(seed=1,n=16,m=16,batch=2,teacher_std_floor=None)
    baseline=GMM40Learner(Target(),**kwargs)
    candidate=GMM40Learner(Target(),hidden_init=initializer,**kwargs)
    old,new=baseline.state.params,candidate.state.params
    for head in ('mu','log_std'):
        for name in ('kernel','bias'):
            np.testing.assert_array_equal(old[head][name],new[head][name])
    for name in ('Dense_0','Dense_1','Dense_2'):
        w=np.asarray(new[name]['kernel'])
        gram=w@w.T if w.shape[0]<=w.shape[1] else w.T@w
        np.testing.assert_allclose(gram,gain**2*np.eye(len(gram)),atol=3e-6)
        np.testing.assert_array_equal(new[name]['bias'],0)
        assert not np.array_equal(old[name]['kernel'],w)
    _,ls=candidate.actor.apply({'params':new},jnp.zeros((16,1)),jnp.ones((16,2)))
    np.testing.assert_array_equal(ls,np.full((16,2),-4.))
    info=candidate.advance(2)
    assert all(np.isfinite(v) for v in info.values())
    assert np.exp(-5)-1e-7 <= info['sigma_min'] <= info['sigma_max'] <= np.exp(-3.5)+1e-7
