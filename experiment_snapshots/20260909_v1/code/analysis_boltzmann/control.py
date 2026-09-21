"""Paired fixed-replay control: common critic/OT actor, only backup changes.

Single scalar Q is exposed as identical twins to the production actor update.
Actor extraction uses the target Q that is evaluated in every backup.
No TD noise; this is deliberately NOT the default OptiQ algorithm.
"""
import argparse,json,time
import numpy as np
import jax,jax.numpy as jnp,optax
from flax import serialization
from common.type_aliases import RLTrainState
from .shared import actor_state,ot_update,begin
from .movecar import Critic,reward,dynamics


def run(args):
    # Controls only: accurate float32 GEMMs; production MoveCar stays unchanged.
    jax.config.update('jax_default_matmul_precision','highest')
    out=begin(args.out,args); rng=np.random.default_rng(args.seed+500000)
    (out/'numeric_protocol.json').write_text(json.dumps({'training_dtype':'float32','matmul_precision':'highest','training_grid':args.grid,'reference_dtype':'float64','reference_tolerance':1e-4,'consecutive_refinements':2,'gate_tolerance':1e-3},indent=2))
    s=rng.uniform(0,10,(65536,1)).astype('float32'); a=rng.uniform(-1,1,(65536,1)).astype('float32')
    sp=dynamics(s,a); r=reward(sp)[:,0].astype('float32')
    np.savez_compressed(out/'replay.npz',s=s,a=a,r=r,sp=sp)
    net=Critic(); params=net.init(jax.random.PRNGKey(args.seed+2),jnp.zeros((1,1)),jnp.zeros((1,1)))['params']
    def twinapply(variables,s,a,**kwargs):
        q=net.apply({'params':variables['params']},s,a)
        return jnp.stack((q,q))[...,None]
    critic=RLTrainState.create(apply_fn=twinapply,params=params,target_params=params,batch_stats={},target_batch_stats={},tx=optax.adam(3e-4))
    actor=actor_state(1,args.seed); key=jax.random.PRNGKey(args.seed+500)
    grid=jnp.linspace(-1+1/args.grid,1-1/args.grid,args.grid)
    @jax.jit
    def update(q,actor,batch,key):
        s,a,r,sp=batch; p=q.target_params
        actions=jnp.broadcast_to(grid[None,:,None],(len(s),len(grid),1))
        obs=jnp.broadcast_to(sp[:,None,:],actions.shape)
        values=net.apply({'params':p},obs,actions)
        if args.backup=='max': v=values.max(1)
        elif args.backup=='boltzmann': v=jnp.sum(jax.nn.softmax(values/.25,axis=1)*values,axis=1)
        elif args.backup=='actor':
            action=jnp.clip(actor.apply_fn({'params':actor.params},sp,jax.random.normal(key,sp.shape)),-1,1)
            v=net.apply({'params':p},sp,action)
        else:
            # Give the local estimator an oracle global-max center, favoring it.
            center=grid[jnp.argmax(values,axis=1)][:,None,None]
            action=jnp.clip(center+jnp.clip(.2*jax.random.normal(key,(len(s),50,1)),-.5,.5),-1,1)
            qs=net.apply({'params':p},jnp.broadcast_to(sp[:,None,:],action.shape),action)
            v=jnp.sum(jax.nn.softmax(qs/.25,axis=1)*qs,axis=1)
        target=jax.lax.stop_gradient(r+.99*v)
        def loss(params): return jnp.mean((net.apply({'params':params},s,a)-target)**2)
        value,grad=jax.value_and_grad(loss)(q.params); q=q.apply_gradients(grads=grad)
        q=q.replace(target_params=jax.tree.map(lambda t,p:.995*t+.005*p,q.target_params,q.params))
        return q,value
    records=[]; start=time.monotonic()
    probe=jnp.repeat(jnp.linspace(0,10,256)[:,None],32,0)
    z=jax.random.normal(jax.random.PRNGKey(args.seed+30000),(len(probe),1))
    # Same initialization/path across control arms; no cross-architecture interpolation.
    initial=actor.params
    def save(step):
        (out/f'checkpoint_{step}.msgpack').write_bytes(serialization.to_bytes({'actor':actor,'critic':critic}))
        from jax.flatten_util import ravel_pytree
        x0,unravel=ravel_pytree(initial); x1,_=ravel_pytree(actor.params)
        @jax.jit
        def objective(x):
            a=jnp.clip(actor.apply_fn({'params':unravel(x)},probe,z),-1,1)
            return -jnp.mean(net.apply({'params':critic.params},probe,a))
        alpha=np.linspace(0,1,101); path=np.array([float(objective((1-t)*x0+t*x1)) for t in alpha])
        d=np.random.default_rng(args.seed+40000).normal(size=(256,len(x1))).astype('float32'); d/=np.linalg.norm(d,axis=1,keepdims=True)
        base=float(objective(x1)); norm=float(jnp.linalg.norm(x1))
        delta=np.array([[float(objective(x1+.05*norm*v))-base,float(objective(x1-.05*norm*v))-base] for v in d])
        np.savez_compressed(out/f'landscape_{step}.npz',alpha=alpha,path=path,perturbation=delta)
        print(json.dumps({'step':step,'backup':args.backup,'elapsed':time.monotonic()-start}),flush=True)
    save(0)
    for step in range(1,args.updates+1):
        ids=rng.integers(len(s),size=256); batch=tuple(jnp.asarray(x[ids]) for x in (s,a,r,sp))
        key,qk,ak=jax.random.split(key,3); critic,loss=update(critic,actor,batch,qk)
        extraction=critic.replace(params=critic.target_params)
        actor,aloss,_,metrics=ot_update(actor,extraction,batch[0],ak)
        if step%100==0:
            records.append({'step':step,'critic_loss':float(loss),'actor_loss':float(aloss),'elapsed':time.monotonic()-start})
            if not np.isfinite([float(loss),float(aloss)]).all(): raise FloatingPointError('loss')
        if step in (100,1000,5000,args.updates): save(step)
    from .control_precision import audit,install_validation
    audit(out,out/'quadrature')
    install_validation(out,out/'quadrature')
    (out/'training.json').write_text(json.dumps(records,indent=2)); (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--backup',choices=['max','boltzmann','actor','local'],required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--updates',type=int,default=20000); ap.add_argument('--grid',type=int,default=2056); ap.add_argument('--out',required=True)
    run(ap.parse_args())
