"""Same oracle and optimizer as screen; enable the explicitly approved large L."""
import jax.numpy as jnp
from ..kl_mode_missing_search_1d.core import Experiment as Base
from ..kl_diverse_targets_1d.core import Experiment as GMM
from ..kl_nongmm_targets_1d.core import Experiment as NonGMM
from ..kl_nongmm_targets_1d.target import logs_and_weights

class GMMExperiment(Base):
    log_terms=GMM.log_terms
    log_f=GMM.log_f
    target_score=GMM.target_score
    def __init__(self,cfg,method,L,seed):
        assert cfg['allow_large_L'] and ((method=='forward' and L==0) or (method=='reverse' and L in (1024,1048576)))
        super().__init__(cfg,method,L,seed)
        self.widths=jnp.asarray(cfg['target_widths']);self.log_masses=jnp.log(jnp.asarray(cfg['target_masses']))

class NonGMMExperiment(Base):
    log_f=NonGMM.log_f
    target_score=NonGMM.target_score
    def __init__(self,cfg,method,L,seed):
        assert cfg['allow_large_L'] and ((method=='forward' and L==0) or (method=='reverse' and L in (1024,1048576)))
        super().__init__(cfg,method,L,seed);self.offset=jnp.asarray(logs_and_weights(cfg))

def implementation(cfg):return NonGMMExperiment if cfg['target_kind']=='nongmm' else GMMExperiment
