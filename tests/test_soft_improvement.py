import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from optiq_dime.soft_improvement import (
    entropy_bracket_sample, conservative_soft_margin, candidate_gap_samples,
)
from optiq_dime.semi_implicit import conditional_mixture_log_prob


def mixture_actor(variables, observations, latents):
    return jnp.where(latents>=0,1.5,-1.5), jnp.full_like(latents,jnp.log(.6))


def test_entropy_bracket_uses_generating_and_independent_components_correctly():
    actor=TrainState.create(apply_fn=mixture_actor,params={'mu':{'bias':jnp.zeros(1)}},tx=optax.sgd(.01))
    obs=jnp.zeros((16384,1))
    _,u,lower,upper=entropy_bracket_sample(actor,obs,jax.random.PRNGKey(22),4)
    mu=jnp.broadcast_to(jnp.array([[[-1.5],[1.5]]]),(len(obs),2,1))
    ls=jnp.full_like(mu,jnp.log(.6))
    true_entropy_samples=-conditional_mixture_log_prob(u[:,None],mu,ls)[:,0]
    assert float(lower.mean()) < float(true_entropy_samples.mean())
    assert float(upper.mean()) > float(true_entropy_samples.mean())
    # Samples themselves are not pointwise entropy bounds.
    assert bool(jnp.any(lower>true_entropy_samples))
    assert bool(jnp.any(upper<true_entropy_samples))


def entropy(p): return -(p*np.log(np.maximum(p,1e-300))).sum(-1)


def exact_value(policy, rewards, transitions, temperature, gamma):
    r=(policy*rewards).sum(-1)+temperature*entropy(policy)
    p=np.einsum('sa,san->sn',policy,transitions)
    return np.linalg.solve(np.eye(len(policy))-gamma*p,r)


def test_statewise_soft_margin_implies_improvement_in_exact_finite_mdp():
    rng=np.random.default_rng(912)
    transitions=rng.dirichlet(np.ones(4),size=(4,3))
    rewards=rng.normal(size=(4,3)); old=rng.dirichlet(np.ones(3),size=4)
    temperature=.5; gamma=.95
    v=exact_value(old,rewards,transitions,temperature,gamma)
    q=rewards+gamma*np.einsum('san,n->sa',transitions,v)
    logits=q/temperature; candidate=np.exp(logits-logits.max(-1,keepdims=True))
    candidate/=candidate.sum(-1,keepdims=True)
    q_gain=((candidate-old)*q).sum(-1)
    margin=conservative_soft_margin(q_gain,entropy(candidate),entropy(old),temperature,
        critic_error_bound=0.,sampling_error_bound=0.)
    assert (margin>=-1e-12).all()
    assert (exact_value(candidate,rewards,transitions,temperature,gamma)>=v-1e-10).all()


def test_flat_q_collapse_rejected_and_critic_error_is_not_ignored():
    margin=conservative_soft_margin(0.,0.,np.log(3),.5,
        critic_error_bound=0.,sampling_error_bound=0.)
    assert margin<0
    naive=conservative_soft_margin(.1,1.,1.,.5,critic_error_bound=0.,sampling_error_bound=0.)
    conservative=conservative_soft_margin(.1,1.,1.,.5,critic_error_bound=.06,sampling_error_bound=0.)
    assert naive>0 and conservative<0
    with pytest.raises(TypeError):
        conservative_soft_margin(.1,1.,1.,.5)  # Learned-Q error cannot silently default to zero.


def test_paired_candidate_check_detects_entropy_improvement_without_q_gradient():
    from common.type_aliases import RLTrainState
    def gaussian(variables,observations,latents):
        params=variables['params']
        return jnp.broadcast_to(params['mu']['bias'],latents.shape), jnp.broadcast_to(params['ls'],latents.shape)
    def q_fn(variables,observations,actions,**kwargs):
        return jnp.broadcast_to(variables['params']['q'][:,None,None],(2,len(actions),1))
    old=TrainState.create(apply_fn=gaussian,
        params={'mu':{'bias':jnp.zeros(2)},'ls':jnp.full((2,),jnp.log(.5))},tx=optax.sgd(.01))
    new=old.replace(params={'mu':{'bias':jnp.zeros(2)},'ls':jnp.full((2,),jnp.log(.9))})
    q=RLTrainState.create(apply_fn=q_fn,params={'q':jnp.zeros(2)},target_params={'q':jnp.zeros(2)},
        batch_stats={},target_batch_stats={},tx=optax.sgd(.01))
    obs=jnp.zeros((32,3)); key=jax.random.PRNGKey(37)
    identical=candidate_gap_samples(old,old,q,obs,key,.5,jnp.zeros(1),4,64)
    np.testing.assert_allclose(identical,0.,atol=1e-6)
    improved=candidate_gap_samples(old,new,q,obs,key,.5,jnp.zeros(1),4,64)
    assert improved.shape==(64,32) and float(improved.mean())>0.1
    for g in jax.tree_util.tree_leaves(jax.grad(lambda params:
        candidate_gap_samples(old,new,q.replace(params=params),obs,key,.5,jnp.zeros(1),4,8).mean())(q.params)):
        np.testing.assert_array_equal(g,jnp.zeros_like(g))
