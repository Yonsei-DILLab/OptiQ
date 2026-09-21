"""Count actual network parameters without copying accelerator tensors to CPU."""
from collections.abc import Mapping
from math import prod
from .evaluation import atomic_json


def count_parameters(value):
    if hasattr(value,'parameters'):return sum(p.numel() for p in value.parameters())
    if hasattr(value,'params'):return count_parameters(value.params)
    if isinstance(value,Mapping):return sum(count_parameters(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(count_parameters(v) for v in value)
    return prod(value.shape)


def save_sizes(folder,**networks):
    counts={key:int(count_parameters(value)) for key,value in networks.items()}
    result=dict(network_parameters=counts,total=sum(counts.values()),
                scope='Live trainable networks; excludes optimizer state and target-network copies.')
    atomic_json(folder/'model_sizes.json',result)
    return result


def fixed_sizes(folder,method,agent):
    if method=='mfpo':return save_sizes(folder,actor=agent.state,divergence=agent.divstate)
    if method in ('optiq','optiq_trg','sql'):return save_sizes(folder,actor=agent.state)
    return save_sizes(folder,**{'joint_flow_QV' if method=='meow' else 'actor':agent.actor})
