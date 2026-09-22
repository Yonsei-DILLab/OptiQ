import numpy as np
from .evaluate import responsibility_mass,sliced_w2
from .adapters import EnergyOnly

def test_soft_component_mass_and_remote_samples():
    means=np.array([[-5.,0.],[5.,0.]])
    x=np.repeat(means,[300,100],axis=0)
    w,near=responsibility_mass(x,means,np.ones(2))
    np.testing.assert_allclose(w,[.75,.25],atol=1e-10)
    assert near.all()
    far,_=responsibility_mass(np.array([[1e4,0.]]),means,np.ones(2))
    np.testing.assert_allclose(far.sum(),1.)

def test_swd_identity_permutation_translation():
    rng=np.random.default_rng(1);x=rng.normal(size=(512,2))
    assert sliced_w2(x,x[::-1])<1e-14  # BLAS summation ordering, float64 roundoff
    dirs=np.random.default_rng(451).normal(size=(128,2));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    v=np.array([1.,2.])
    np.testing.assert_allclose(sliced_w2(x,x+v),np.sqrt(np.mean((dirs@v)**2)),rtol=1e-12)

def test_oracle_interface():
    class Target:
        means='secret'
        def sample(self,*a):raise AssertionError('reference leaked')
        def jax_log_prob(self,x):return x
        def torch_log_prob(self,x):return x
    oracle=EnergyOnly(Target())
    assert not hasattr(oracle,'sample') and not hasattr(oracle,'means')
