"""Two independent scalar Q networks used by the gmm-trg configuration."""
from collections.abc import Sequence
import flax.linen as nn
import jax.numpy as jnp

class Critic(nn.Module):
    net_arch: Sequence[int]
    @nn.compact
    def __call__(self, obs, action, train=True):
        x = jnp.concatenate((obs,action),axis=-1)
        for width in self.net_arch:
            x = nn.gelu(nn.Dense(width)(x))
        return nn.Dense(1)(x)

class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    n_critics: int = 2
    @nn.compact
    def __call__(self, obs, action, train=True):
        ensemble = nn.vmap(Critic,variable_axes={'params':0,'batch_stats':0},
            split_rngs={'params':True,'dropout':True,'batch_stats':True},
            in_axes=None,out_axes=0,axis_size=self.n_critics)
        return ensemble(net_arch=self.net_arch)(obs,action,train)
