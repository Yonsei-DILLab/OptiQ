"""Independent numerical oracles and regression cases for invalid labels."""
import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import torch
from verified_solver import Mixture,ascend,catalog,assign,verify,filtration


class VerificationTests(unittest.TestCase):
    def test_density_derivatives_and_normalization(self):
        mix=Mixture([[-.35,.1],[.4,-.25]],np.log([[.22,.4],[.3,.18]]))
        x=torch.tensor([.13,-.08],dtype=torch.float64,requires_grad=True)
        def direct(xx):
            return torch.logsumexp(mix.log_norm-.5*(((xx-mix.mu)/mix.sd)**2).sum(-1),0)
        g=torch.autograd.functional.jacobian(direct,x)
        h=torch.autograd.functional.hessian(direct,x)
        gg,hh,_=mix.derivatives(x.detach()[None])
        torch.testing.assert_close(mix.logp(x[None])[0],direct(x),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(gg[0],g,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(hh[0],h,atol=1e-9,rtol=1e-9)
        m=Mixture([[-.7],[.5]],np.log([[.3],[.2]]))
        integral=quad(lambda v:math.exp(float(m.logp(torch.tensor([[v]],dtype=torch.float64))[0])),-1,1,epsabs=1e-10)[0]
        self.assertAlmostEqual(integral,1,places=9)
        # D>1 is essential: the legacy normalization multiplied log(N) by D.
        single=Mixture([[.2,-.3]],np.log([[.12,.25]]))
        repeated=Mixture(np.repeat([[.2,-.3]],3,axis=0),np.log(np.repeat([[.12,.25]],3,axis=0)))
        point=torch.tensor([[.1,.2],[-.5,.3]],dtype=torch.float64)
        torch.testing.assert_close(single.logp(point),repeated.logp(point),atol=1e-11,rtol=1e-11)

    def test_unimodal_and_reject_nonstationary(self):
        mix=Mixture([[.2,-.3]],np.log([[.12,.25]]))
        x=torch.tensor([[-.9,.7],[.8,.5],[0.,0.]],dtype=torch.float64)
        end,_,info=ascend(mix,x);m,_=catalog(mix,end);labels,assigned=assign(mix,end,m)
        self.assertEqual(info['unconverged'],0);self.assertEqual(len(m),1);self.assertEqual(assigned['far'],0)
        self.assertFalse(bool(verify(mix,x[:1])[0][0])) # negative Hessian alone is insufficient
        with self.assertRaises(RuntimeError):assign(mix,end,m+.02)

    def test_bimodal_against_independent_1d_root(self):
        means=np.array([[-.45],[.45]]);sd=.16
        mix=Mixture(means,np.log(np.full_like(means,sd)))
        # Equal truncated normalizers cancel; this is an independent direct score.
        def score(t):
            w=np.exp(-.5*((t-means[:,0])/sd)**2)
            return np.sum(w*(means[:,0]-t))/sd**2/np.sum(w)
        peak=brentq(score,.1,.7,xtol=1e-14)
        start=torch.linspace(-.99,.99,100,dtype=torch.float64)[:,None]
        end,_,info=ascend(mix,start);m,_=catalog(mix,end)
        self.assertEqual(info['unconverged'],0);self.assertEqual(len(m),2)
        np.testing.assert_allclose(np.sort(m[:,0].numpy()),[-peak,peak],atol=1e-9)
        b=mix.logp(m).numpy();saddle=float(mix.logp(torch.zeros((1,1),dtype=torch.float64))[0])
        pairs,_,roots=filtration(b,np.array([[0,1]]),np.array([saddle]))
        self.assertEqual(len(roots),1);self.assertGreater(pairs[0,0]-pairs[0,1],2)
        self.assertFalse(bool(verify(mix,torch.zeros((1,1),dtype=torch.float64))[0][0]))

    def test_separable_sac_filtration(self):
        # Two independent tanh N(0,1) coordinates have four identical peaks.
        u=brentq(lambda z:z-2*np.tanh(z),.1,3.)
        def lp(z):return -.5*z*z+2*np.log(np.cosh(z))-.5*np.log(2*np.pi)
        birth=2*lp(u);saddle=lp(u)+lp(0.)
        edge=np.array([[0,1],[0,2],[1,3],[2,3]])
        pairs,_,roots=filtration(np.full(4,birth),edge,np.full(4,saddle))
        np.testing.assert_allclose(pairs[:,0]-pairs[:,1],lp(u)-lp(0.),atol=1e-13)
        self.assertEqual(len(pairs),3);self.assertEqual(len(roots),1)
        with self.assertRaises(ValueError):filtration(np.array([1.,2.]),np.array([[0,1]]),np.array([1.1]))

    def test_endpoint_cache_does_not_leak_larger_sample_modes(self):
        from run_verified import execute
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'mix.npz';samples=root/'samples.npy'
            np.savez(data,mu=np.array([[[-.45],[.45]]]),log_std=np.full((1,2,1),np.log(.16)))
            np.save(samples,np.linspace(-.9,.9,64)[:,None])
            base=dict(data=data,samples=samples,device='cpu',maxit=20000,k=15,commit='unit-test',
                      upstream=Path(__file__).with_name('upstream_basin_ph.py'))
            for name,limit,cache in [('full',0,None),('direct',32,None),('cached',32,root/'full'/'endpoints.npz')]:
                out=root/name;out.mkdir();execute(SimpleNamespace(**base,out=out,limit=limit,endpoints=cache))
            with np.load(root/'full'/'verified_graph.npz') as f:self.assertEqual(len(f['modes']),2)
            with np.load(root/'direct'/'verified_graph.npz') as f,np.load(root/'cached'/'verified_graph.npz') as c:
                self.assertEqual(len(c['modes']),1)
                for key in ('modes','births','pairs','labels'):np.testing.assert_allclose(f[key],c[key],atol=1e-7)
            wrong=root/'wrong.npz';np.savez(wrong,points=np.zeros((64,1)),data_sha256='wrong',samples_sha256='wrong')
            out=root/'invalid';out.mkdir()
            with self.assertRaises(ValueError):execute(SimpleNamespace(**base,out=out,limit=32,endpoints=wrong))

    def test_nearby_approximations_polish_to_one_root(self):
        # Both endpoints meet a scaled 1e-8 gradient tolerance, yet their
        # coordinate separation exceeds the equality tolerance. They are one root.
        mix=Mixture([[0.]],[[np.log(100.)]])
        endpoints=torch.tensor([[-8e-8],[8e-8]],dtype=torch.float64)
        self.assertTrue(bool(verify(mix,endpoints)[0].all()))
        modes,info=catalog(mix,endpoints)
        self.assertEqual(len(modes),1);self.assertLess(abs(float(modes[0,0])),1e-12)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
