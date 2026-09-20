"""Synchronous data parallelism for large fixed-Q OptiQ sample counts.

Each device solves independent N x M OT or Direct GMM problems. Gradients are
averaged before one shared Adam step; N, M, effective batch and objective are
unchanged by sharding. Each replica has its own reproducible RNG stream.
"""
import flax.serialization
import jax
import numpy as np

from .optiq import OptiQ


class ParallelOptiQ(OptiQ):
    def __init__(self,target,seed=0,n=16,m=64,batch=256,epsilon=.1,hidden_dims=(256,256),sinkhorn_iterations=100,
                 mean_output_init_scale=1e-4,temperature=1.,nll_top_k=None,nll_plan_threshold=None,devices=4,
                 sigma_row_balance=False,distillation_loss="conditional_ot_nll"):
        if devices<2 or batch%devices:
            raise ValueError('Parallel OptiQ requires >=2 devices and an evenly divisible global batch')
        available=jax.local_devices()
        if len(available)<devices:raise ValueError(f'Requested {devices} devices, available {len(available)}')
        self.parallel_devices=available[:devices]
        self.device_count=devices;self.global_batch=batch
        super().__init__(target,seed,n,m,batch//devices,epsilon,hidden_dims,sinkhorn_iterations,
                         mean_output_init_scale,temperature,nll_top_k,nll_plan_threshold,sigma_row_balance,distillation_loss)
        self.gradient_axis='optiq_batch'
        self.parallel_state=jax.device_put_replicated(self.state,self.parallel_devices)
        # Shape-independent per-replica streams; no claim of matching an
        # unsharded random tensor bit-for-bit.
        keys=[jax.random.fold_in(self.key,i) for i in range(devices)]
        self.parallel_keys=jax.device_put_sharded(keys,self.parallel_devices)
        self.parallel_advance=jax.pmap(self._advance,axis_name=self.gradient_axis,
                                      devices=self.parallel_devices,static_broadcasted_argnums=(2,))

    def advance(self,count):
        self.parallel_state,self.parallel_keys,info=self.parallel_advance(self.parallel_state,self.parallel_keys,count)
        info={k:float(np.asarray(v)[0]) for k,v in info.items()}
        self.state=jax.tree_util.tree_map(lambda x:x[0],self.parallel_state)
        self.key=self.parallel_keys[0]
        self.updates+=count
        info['Q_evaluations']=self.updates*self.global_batch*self.m
        return info

    def save(self,path):
        path.write_bytes(flax.serialization.to_bytes(dict(state=self.state,key=self.key,updates=self.updates,
            parallel_keys=np.asarray(self.parallel_keys),device_count=self.device_count,global_batch=self.global_batch)))

    def restore(self,path):
        template=dict(state=self.state,key=self.key,updates=0,parallel_keys=np.asarray(self.parallel_keys),
                      device_count=self.device_count,global_batch=self.global_batch)
        saved=flax.serialization.from_bytes(template,path.read_bytes())
        if saved['device_count']!=self.device_count or saved['global_batch']!=self.global_batch:
            raise ValueError('Resume must preserve device count and global batch')
        self.state,self.key,self.updates=saved['state'],saved['key'],int(saved['updates'])
        self.parallel_state=jax.device_put_replicated(self.state,self.parallel_devices)
        self.parallel_keys=jax.device_put_sharded(list(saved['parallel_keys']),self.parallel_devices)
