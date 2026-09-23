import json,subprocess
from pathlib import Path
import jax,numpy as np
from scipy.integrate import quad
from .core import Experiment,CENTERS,WIDTH,q_value
from .evaluate import cdf,target_logf,separation,Z,REFERENCE_BACKUP
from . import box_gaussian as new_box
from ..kl_forward_far_1d import box_gaussian as old_box
from ..kl_forward_far_1d.actor import SemiImplicitActor as OldActor

def main():
    c=json.loads(Path(__file__).with_name('config.json').read_text());e=Experiment(c,'forward',0,0)
    assert CENTERS==(-5.,0.,5.) and WIDTH==1 and c['n']==c['m']==128 and c['steps']==100000
    f='experiments/kl_forward_far_1d/distillation.py'
    assert subprocess.check_output(['git','show','1153ff9a797173681a980fffea6f2e75414fe40d:'+f])==Path(__file__).with_name('distillation.py').read_bytes()
    z=jax.random.normal(jax.random.PRNGKey(1),(128,1))
    mu,ls=e.components(e.state.params,z)
    old_actor=OldActor(1,(256,256),-5.,-1.,-1.,1.)
    old_mu,old_ls=old_actor.apply({'params':e.state.params},jax.numpy.zeros_like(z),z)
    assert np.array_equal(mu,old_mu/2) and np.array_equal(ls,old_ls)
    means=jax.numpy.array([[-10.],[-9.99],[0.],[9.99],[10.]])
    logs=jax.numpy.full_like(means,-1.);key=jax.random.PRNGKey(11)
    aa=new_box.sample_box(key,means,logs)
    bb=old_box.sample_box(key,2*means,logs+jax.numpy.log(2.))/2
    assert np.allclose(aa,bb,atol=2e-6) and np.max(np.abs(aa))<=10
    lp=new_box.component_log_prob(aa[None],means[None],logs[None])
    lp0=old_box.component_log_prob((2*aa)[None],(2*means)[None],(logs+jax.numpy.log(2.))[None])+jax.numpy.log(2.)
    assert np.allclose(lp,lp0,atol=2e-3,rtol=2e-6)
    grid=np.linspace(-10,10,40001);density=np.exp(target_logf(grid))/Z
    peaks=grid[np.where((density[1:-1]>density[:-2])&(density[1:-1]>density[2:]))[0]+1]
    assert len(peaks)==3 and np.allclose(peaks,CENTERS,atol=.002)
    integral=quad(lambda x:float(np.exp(target_logf(x))/Z),-10,10,points=list(CENTERS),epsabs=1e-11)[0]
    assert abs(integral-1)<1e-10 and abs(cdf(10)-cdf(-10)-1)<1e-12
    actions=np.interp((np.arange(32768)+.5)/32768,cdf(grid),grid)
    assert separation(actions)['three_peak_pass']
    assert not separation(np.linspace(-10,10,32768))['three_peak_pass']
    assert np.allclose(np.asarray(q_value(jax.numpy.asarray(grid[:,None]))),.25*target_logf(grid),atol=2e-5)
    (loss,metrics),grad=jax.value_and_grad(e.group_loss,has_aux=True)(e.state.params,jax.random.PRNGKey(87))
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(grad))
    print(json.dumps(dict(passed=True,total_updates=c['steps'],action_bound=c['action_bound'],target_integral=integral,target_retained_probability=Z,peak_positions=peaks.tolist(),target_diagnostic=separation(actions),reference_backup=REFERENCE_BACKUP,loss=float(loss),mean_half_of_prior=True,sigma_unchanged=True,sampler_coordinate_identity_error=float(np.max(np.abs(aa-bb))),density_coordinate_identity_error=float(np.max(np.abs(lp-lp0))),marginal_NLL_unchanged=True),indent=2))

if __name__=='__main__':main()
