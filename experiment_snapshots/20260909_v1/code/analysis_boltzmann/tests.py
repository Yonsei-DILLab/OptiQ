"""Standalone numerical tests: avoids repository conftest's online W&B fixture."""
import unittest
import numpy as np
from scipy.special import logsumexp
from .problems import make_problem,local_estimates

class NumericalTests(unittest.TestCase):
    def test_constant(self):
        p=make_problem('constant'); self.assertAlmostEqual(p.truth()[2],.7)
        for m in ('no_is','official_is','truncated_is'):
            v,a,w=local_estimates(p,np.array([.9]),50,100,np.random.default_rng(7),m)
            np.testing.assert_allclose(v,.7,atol=1e-12); self.assertTrue(np.all(abs(a)<=1)); np.testing.assert_allclose(w.sum(1),1)
    def test_separable_truth(self):
        p=make_problem('separable_4'); self.assertAlmostEqual(p.truth()[2],4*p.base().truth()[2],places=10)
    def test_references(self):
        for name in ('unimodal','symmetric_1.3_0.12','asymmetric_1.3_0.08','modes2d_4','modes2d_8'):
            p=make_problem(name); g,m,v,n,delta=p.truth()
            self.assertLess(delta,1e-4); self.assertTrue(np.isfinite(v)); self.assertAlmostEqual(m.sum(),1)
    def test_is_restricted_truth(self):
        p=make_problem('asymmetric_1.3_0.08'); center=np.array([-.65]); rng=np.random.default_rng(42)
        a=np.linspace(-1,-.15,20000)[:,None]; q=p.q(a); w=np.exp(q/.25-logsumexp(q/.25)); truth=w@q
        v,_,_=local_estimates(p,center,4096,200,rng,'truncated_is')
        self.assertLess(abs(v.mean()-truth),.015)
        self.assertGreater(abs(truth-p.truth()[2]),.05)
    def test_k1(self):
        p=make_problem('unimodal')
        for mode in ('no_is','official_is','truncated_is'):
            v,a,_=local_estimates(p,np.array([0.]),1,100,np.random.default_rng(5),mode)
            np.testing.assert_allclose(v,p.q(a)[:,0])
    def test_environment(self):
        from .movecar import MoveCar
        e=MoveCar(); s,_=e.reset(); self.assertEqual(s[0],8)
        s,r,terminal,trunc,_=e.step(np.array([1.])); self.assertEqual(r,1); self.assertFalse(terminal)
        for _ in range(99): s,r,terminal,trunc,_=e.step(np.array([0.]))
        self.assertTrue(trunc); self.assertFalse(terminal)
    def test_jax_energy(self):
        import jax.numpy as jnp,jax.scipy as jsp
        for name in ('asymmetric_1.3_0.08','separable_4','modes2d_8'):
            p=make_problem(name); a=np.random.default_rng(4).uniform(-1,1,(50,p.dim)).astype('float32')
            np.testing.assert_allclose(p.q(a),np.asarray(p.q(jnp.asarray(a),jnp,jsp.special.logsumexp)),rtol=1e-5,atol=1e-5)

if __name__=='__main__': unittest.main()
