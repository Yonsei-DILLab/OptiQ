"""Only the oracle target changes; inherit the validated TRG update verbatim."""
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from ..kl_mode_missing_search_1d.core import Experiment as OriginalExperiment


class Experiment(OriginalExperiment):
    def __init__(self,cfg,method,L,seed):
        assert method=='forward' and L==0 or method=='reverse' and L==1024, 'High-L confirmation awaits user approval'
        super().__init__(cfg,method,L,seed)
        self.widths=jnp.asarray(cfg['target_widths'])
        self.log_masses=jnp.log(jnp.asarray(cfg['target_masses']))

    def log_terms(self,a):
        x=a[...,0,None]
        return self.log_masses-jnp.log(self.widths)-.5*math.log(2*math.pi)-.5*((x-self.centers)/self.widths)**2

    def log_f(self,a):
        return jsp.special.logsumexp(self.log_terms(a),axis=-1)

    def target_score(self,a):
        r=jax.nn.softmax(self.log_terms(a),axis=-1)
        return (r*(self.centers-a[...,0,None])/self.widths**2).sum(-1)[...,None]
