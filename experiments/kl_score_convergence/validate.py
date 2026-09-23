import json
import numpy as np
import jax
import jax.numpy as jnp
from .measure import make_accumulator,quadrature_nodes,stable_minimum
from experiments.kl_direction_1d.core import dense_score

def main():
    rng=np.random.default_rng(51);a=np.linspace(-.98,.98,81);mu=rng.uniform(-.8,.8,(512,1));ls=rng.uniform(-3,-1,(512,1))
    w=rng.uniform(.01,2,512);zero=(np.full(len(a),-np.inf),np.zeros(len(a)))
    acc=make_accumulator(128);carry=acc(mu,ls,np.log(w),a,0,2,zero)
    split=acc(mu,ls,np.log(w),a,2,4,carry);full=acc(mu,ls,np.log(w),a,0,4,zero)
    assert np.max(np.abs(np.array(split)-np.array(full)))<1e-11
    # Weighted dense reference, including component truncation normalization.
    from experiments.kl_direction_1d.box_gaussian import component_log_prob
    ell=component_log_prob(jnp.asarray(a[None,:,None]),jnp.asarray(mu[None]),jnp.asarray(ls[None]))[0]+jnp.log(w[:,None])
    exact=jax.nn.softmax(ell,axis=0)*((mu-a[None,:])*np.exp(-2*ls))
    assert np.max(np.abs(np.asarray(full[1])-np.asarray(exact.sum(0))))<1e-10
    z,lw=quadrature_nodes(256,16,12);mu0=np.full_like(z,.2);ls0=np.full_like(z,-2)
    lp,score=acc(mu0,ls0,lw,a,0,len(z)//128,zero)
    expected=(float(mu0[0,0])-a)*np.exp(4)
    error=float(np.max(np.abs(np.asarray(score)-expected)));assert error<1e-10
    assert abs(np.exp(lw).sum()-1)<1e-12
    assert stable_minimum([128,256,512],np.array([[.1,.009,.008],[.1,.007,.004]]),np.array([[.04,.008],[.08,.008]]),.01)==256
    assert stable_minimum([128,256],np.ones((2,2)),np.ones((2,1)),.01) is None
    assert np.array_equal(np.random.default_rng(3).standard_normal((128,1)),np.random.default_rng(3).standard_normal((512,1))[:128])
    print(json.dumps(dict(passed=True,weighted_chunk_error=float(np.max(np.abs(np.array(split)-np.array(full)))),constant_component_quadrature_score_error=error)))

if __name__=='__main__':main()
