"""Enable the previously validated reverse path on the unchanged width-1 actor."""
from ..kl_forward_wide_1d.core import Experiment as ForwardExperiment
from ..kl_forward_wide_1d.core import CENTERS, WIDTH, q_value, log_f, target_score, dense_score
from ..kl_forward_wide_1d.box_gaussian import component_log_prob
import jax
import jax.numpy as jnp


class Experiment(ForwardExperiment):
    def __init__(self, cfg, method, L, seed):
        assert method == 'reverse' and int(L) > 0
        assert int(L) % min(cfg['density_chunk'], int(L)) == 0
        # Preserve actor, optimizer, and RNG initialization exactly. JIT methods
        # are first traced after __init__, with the reverse condition below.
        super().__init__(cfg, 'forward', 0, seed)
        self.method, self.L = method, int(L)

    def density_components(self, params, z):
        # Preserve default-precision source actions exactly as in forward; only
        # density-bank MLP evaluation needs consistent full-float32 dot products.
        with jax.default_matmul_precision('highest'):
            return self.components(params, z)

    def density_score(self, params, a, key, L):
        """Stream the SAME IID bank with max-scaled sums, avoiding log cancellation.

        Directly accumulating score numerators and denominators avoids losing
        convex weight normalization when log densities are very negative.
        This is the identical finite-bank ratio estimator, not another objective.
        """
        chunk=min(self.cfg['density_chunk'],L)
        assert L % chunk == 0
        params=jax.tree_util.tree_map(jax.lax.stop_gradient,params)
        a=jax.lax.stop_gradient(a)
        def block(carry,index):
            peak,total,numerator,squared=carry
            z=jax.random.normal(jax.random.fold_in(key,index),(chunk,1))
            mu,ls=self.density_components(params,z)
            ell=component_log_prob(a[None],mu[None],ls[None])[0]
            local_peak=jnp.max(ell,axis=0)
            mass=jnp.exp(ell-local_peak[None])
            local_score=(-(a[None]-mu[:,None])*jnp.exp(-2*ls[:,None]))
            new_peak=jnp.maximum(peak,local_peak)
            old_scale=jnp.exp(peak-new_peak)
            new_scale=jnp.exp(local_peak-new_peak)
            total=old_scale*total+new_scale*mass.sum(0)
            numerator=old_scale[:,None]*numerator+new_scale[:,None]*(mass[...,None]*local_score).sum(0)
            squared=old_scale**2*squared+new_scale**2*(mass**2).sum(0)
            return (new_peak,total,numerator,squared),None
        shape=(a.shape[0],)
        init=(jnp.full(shape,-jnp.inf),jnp.zeros(shape),jnp.zeros_like(a),jnp.zeros(shape))
        (peak,total,numerator,squared),_=jax.lax.scan(block,init,jnp.arange(L//chunk))
        return numerator/total[:,None],peak+jnp.log(total)-jnp.log(L),total**2/squared
