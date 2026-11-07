import jax
import jax.numpy as jnp
import jax.scipy as jsp
from ..kl_mode_missing_search_1d.core import Experiment as OriginalExperiment
from .target import jax_log_shape, logs_and_weights

class Experiment(OriginalExperiment):
    def __init__(self,cfg,method,L,seed):
        assert (method=='forward' and L==0) or (method=='reverse' and L==1024)
        super().__init__(cfg,method,L,seed)
        self.offset=jnp.asarray(logs_and_weights(cfg))

    def log_f(self,a):
        terms=jnp.stack([jax_log_shape(a[...,0],t) for t in self.cfg['shapes']],axis=-1)
        return jsp.special.logsumexp(terms+self.offset,axis=-1)

    def target_score(self,a):return jax.grad(lambda x:self.log_f(x).sum())(a)
