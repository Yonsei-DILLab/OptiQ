import jax
import jax.numpy as jnp
import numpy as np

from optiq_dime.gaussian_transport import (
    fresh_balanced_sinkhorn, source_log_density, gaussian_assignment_log_probs,
)


def test_independent_fresh_state_solves_match_separate_solves_and_query():
    mu = jnp.array([[[-.7],[.4],[1.2]], [[.8],[-1.],[.1]]])
    ls = jnp.log(jnp.array([[[.4],[.7],[.5]], [[.2],[.6],[.9]]]))
    query = jnp.array([[[-.5],[.9]], [[-.3],[1.1]]])
    weights = jnp.array([[.3,.7],[.65,.35]])
    kernel = source_log_density(query,mu,ls).swapaxes(1,2)
    result = fresh_balanced_sinkhorn(kernel,weights,max_iterations=1000,relative_tolerance=1e-4)
    assert np.all(result['converged'])
    np.testing.assert_allclose(result['source_mass'], 1/3, rtol=1e-4)
    np.testing.assert_allclose(result['teacher_mass'], weights, rtol=2e-6)
    for b in range(2):
        separate = fresh_balanced_sinkhorn(kernel[b:b+1],weights[b:b+1],max_iterations=1000,relative_tolerance=1e-4)
        np.testing.assert_allclose(result['plan'][b:b+1],separate['plan'],rtol=1e-5,atol=1e-7)
    log_r = gaussian_assignment_log_probs(query,mu,ls,result['source_potential'])
    np.testing.assert_allclose(jnp.exp(log_r).swapaxes(1,2)*weights[:,None],result['plan'],rtol=2e-5,atol=1e-7)
    assert not np.allclose(result['source_potential'][0],result['source_potential'][1])


def test_query_gradient_is_live_and_scale_jacobian_cancels_assignment():
    mu=jnp.array([[[-1.],[1.]]]); ls=jnp.log(jnp.ones_like(mu)*.7)
    potential=jnp.array([[.2,-.3]]); a=jnp.array([[[.1]]])
    f=lambda x:gaussian_assignment_log_probs(x,mu,ls,potential)[0,0,0]
    gradient=jax.grad(f)(a)
    finite_difference=(f(a+1e-3)-f(a-1e-3))/(2e-3)
    np.testing.assert_allclose(gradient.squeeze(),finite_difference,rtol=1e-3)
    assert abs(float(gradient.squeeze()))>.1
    np.testing.assert_allclose(gaussian_assignment_log_probs(a,mu,ls,potential,40.),
                               gaussian_assignment_log_probs(a,mu,ls,potential,1.),atol=1e-6)


def test_unconverged_plan_is_not_labeled_balanced():
    kernel=jnp.array([[[0.,-8.],[-8.,0.]]])
    result=fresh_balanced_sinkhorn(kernel,jnp.array([[.9,.1]]),min_iterations=1,max_iterations=1,relative_tolerance=1e-4)
    assert not bool(result['converged'][0])
    assert float(result['row_relative_error'][0])>.1


def test_roundoff_at_tolerance_keeps_iterating_until_returned_plan_is_balanced():
    # Portable small version of the initial GPU campaign's stopping mismatch.
    # With the old exp(kernel+f+g) criterion on CPU, this stopped after one
    # iteration but returned row error 0.001000046730041504 > the same 1e-3
    # tolerance. The actual softmax-normalized plan needs a second iteration.
    # Some backends round the first step to the other side of the threshold;
    # either way, an early exit must return a converged plan.
    kernel=jnp.array([[[0.,0.],[-.1265944540500641,0.]]],dtype=jnp.float32)
    weights=jnp.full((1,2),.5,dtype=jnp.float32)
    one=fresh_balanced_sinkhorn(kernel,weights,min_iterations=1,max_iterations=1,
                               relative_tolerance=1e-3)
    result=fresh_balanced_sinkhorn(kernel,weights,min_iterations=1,max_iterations=500,
                                  relative_tolerance=1e-3)
    assert bool(result['converged'][0])
    assert int(result['iterations'][0])<500
    if not bool(one['converged'][0]):
        assert int(result['iterations'][0])>1
    assert float(result['row_relative_error'][0])<=1e-3
    assert float(result['column_relative_error'][0])<=1e-3
    np.testing.assert_allclose(result['source_mass'],.5,rtol=1e-3,atol=0.)
