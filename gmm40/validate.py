"""Numerical invariants that matter for the cross-method comparison."""
import json
import math
import sys

import numpy as np
from .target import Target,ROOT,RESULTS,initialize_target


def main():
    import torch
    import jax
    import jax.numpy as jnp
    target=initialize_target()
    rng=np.random.default_rng(937)
    points=rng.uniform(-40,40,size=(300,2)).astype(np.float32)
    expected=target.log_prob(points)
    actual=target.torch_log_prob(torch.tensor(points)).numpy()
    jax_actual=np.asarray(target.jax_log_prob(jnp.asarray(points)))
    assert np.allclose(actual,expected,atol=1e-4,rtol=1e-5)
    assert np.allclose(jax_actual,expected,atol=1e-4,rtol=1e-5)
    # Compare differentiable Q gradient to an independent float64 finite difference.
    selected=torch.tensor(points[:20],requires_grad=True)
    grad=torch.autograd.grad(target.torch_log_prob(selected).sum(),selected)[0].numpy()
    numerical=np.empty_like(grad)
    for d in range(2):
        plus=points[:20].astype(np.float64);minus=plus.copy()
        plus[:,d]+=.001;minus[:,d]-=.001
        numerical[:,d]=(target.log_prob(plus)-target.log_prob(minus))/.002
    assert np.allclose(grad,numerical,atol=2e-4,rtol=2e-4)
    # MEow's physical-density Q-V identity, including scale 40.
    sys.path.insert(0,str(ROOT/"gmm40-baseline/meow/toy"))
    from modules.policy import FlowPolicy
    torch.manual_seed(0)
    model=FlowPolicy(1.,1.,-2.,2,1,"cpu").eval()
    actions=torch.tensor(points[:40]/45)
    actions=torch.cat([actions,actions]);obs=torch.zeros((80,1))
    with torch.no_grad():
        q,v=model.get_qv(obs,actions)
        logp=model.log_prob(obs,actions)-2*math.log(40)
    identity_error=float((q[:,0]-2*math.log(40)-v[:,0]-logp).abs().max())
    assert identity_error<2e-5
    # The exact same v5 NLL helper is used; compare to direct full tensor likelihood.
    from optiq_dime.distillation import conditional_ot_nll
    mu=jnp.asarray(rng.normal(size=(3,4,2)));ls=jnp.asarray(rng.normal(size=(3,4,2))*.2)
    teacher=jnp.asarray(rng.normal(size=(3,7,2)))
    rows=jax.nn.softmax(jnp.asarray(rng.normal(size=(3,4,7))),axis=-1)
    direct=.5*jnp.sum(((teacher[:,None,:,:]-mu[:,:,None,:])/jnp.exp(ls[:,:,None,:]))**2+2*ls[:,:,None,:]+math.log(2*math.pi),axis=-1)
    expected_nll=jnp.mean(jnp.sum(rows*direct,axis=-1))
    actual_nll=conditional_ot_nll(mu,ls,teacher,rows)
    assert np.allclose(actual_nll,expected_nll,atol=1e-6)
    from .navigation import GMM40Navigation,movement,vector_rollout
    env=GMM40Navigation();s,_=env.reset(seed=97);s2,_=env.reset(seed=97)
    assert np.array_equal(s,s2)
    for i in range(100):
        expected_state=movement(env.state,[.3,.4])
        state,reward,done,trunc,_=env.step([.3,.4])
        assert np.array_equal(state,expected_state) and done==(i==99) and not trunc
        assert abs(reward-float(target.log_prob(state)))<1e-7
    rollout=vector_rollout(lambda state:np.ones_like(state),n=16)
    assert rollout['positions'].shape==(101,16,2)
    assert rollout['actions'].shape==(100,16,2)
    assert np.isfinite(rollout['rewards']).all()
    result=dict(status="passed",torch_numpy_max_error=float(np.max(np.abs(actual-expected))),
                jax_numpy_max_error=float(np.max(np.abs(jax_actual-expected))),
                Q_gradient_max_error=float(np.max(np.abs(grad-numerical))),meow_identity_max_error=identity_error,
                optiq_nll_error=float(abs(actual_nll-expected_nll)),navigation="seeded reset, unit movement, reward, terminal at100, vector trajectory shapes",
                boundary_mass=target.metadata['outside_mass'])
    (RESULTS/"numerical_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
