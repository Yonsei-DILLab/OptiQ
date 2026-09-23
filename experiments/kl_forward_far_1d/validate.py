"""CPU numerical checks; no training sweep or GPU allocation."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from . import box_gaussian as new
from ..kl_direction_1d import box_gaussian as old
from ..kl_direction_1d.actor import SemiImplicitActor as OldActor
from .core import Experiment
from .evaluate import cdf, target_logf, separation, Z, REFERENCE_BACKUP

def main():
    cfg=json.loads(Path(__file__).with_name('config_64.json').read_text())
    e=Experiment(cfg,'forward',0,0)
    previous=OldActor(1,tuple(cfg['hidden_dims']),-5.,-1.,-1.,1.)
    z=jax.random.normal(jax.random.PRNGKey(71),(32768,1))
    mu,ls=e.components(e.state.params,z)
    mu0,ls0=previous.apply({'params':e.state.params},jnp.zeros_like(z),z)
    assert np.allclose(mu,20*mu0) and np.array_equal(ls,ls0)
    init=jax.random.split(jax.random.PRNGKey(0))[1]
    params0=previous.init(init,jnp.zeros((1,1)),jnp.zeros((1,1)))['params']
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(params0),jax.tree_util.tree_leaves(e.state.params)))
    # Near-boundary cases plus interior, sigma unchanged in physical units.
    mus=jnp.array([[-20.],[-19.99],[0.],[19.99],[20.]])
    logs=jnp.full_like(mus,-1.);key=jax.random.PRNGKey(9)
    a=new.sample_box(key,mus,logs)
    a0=20*old.sample_box(key,mus/20,logs-jnp.log(20.))
    assert np.allclose(a,a0,atol=3e-6)
    assert np.all(np.abs(a)<=20)
    ell=new.component_log_prob(a[None],mus[None],logs[None])
    ell0=old.component_log_prob((a/20)[None],(mus/20)[None],(logs-jnp.log(20.))[None])-jnp.log(20.)
    assert np.allclose(ell,ell0,atol=3e-3,rtol=2e-6)
    normalization=[]
    for m in [-20.,-19.99,0.,19.99,20.]:
        sigma=np.exp(-1);lo=max(-20,m-10*sigma);hi=min(20,m+10*sigma)
        from scipy.special import ndtr
        normalizer=ndtr((20-m)/sigma)-ndtr((-20-m)/sigma)
        integral=quad(lambda x:np.exp(-.5*((x-m)/sigma)**2)/(sigma*np.sqrt(2*np.pi)*normalizer),lo,hi,epsabs=1e-11)[0]
        normalization.append(integral)
    assert np.allclose(normalization,1,atol=1e-10)
    (loss,metrics),grad=jax.value_and_grad(e.group_loss,has_aux=True)(e.state.params,jax.random.PRNGKey(2))
    assert np.isfinite(loss) and all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(grad))
    grid=np.linspace(-20,20,262145);target=np.interp((np.arange(32768)+.5)/32768,cdf(grid),grid)
    assert separation(target)['three_peak_pass']
    assert not separation(np.linspace(-20,20,32768))['three_peak_pass']
    assert abs(cdf(20)-cdf(-20)-1)<1e-12 and abs(np.diff(cdf(np.linspace(-20,20,4097))).sum()-1)<1e-12
    result=dict(passed=True,box_sampling_coordinate_identity_max_error=float(np.max(np.abs(a-a0))),density_coordinate_identity_max_error=float(np.max(np.abs(ell-ell0))),component_integrals=normalization,initial_parameter_identity=True,mean_multiplier=20,sigma_unchanged=True,target_Z=Z,reference_backup=REFERENCE_BACKUP,initial_loss=float(loss),target_three_peak=separation(target))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
