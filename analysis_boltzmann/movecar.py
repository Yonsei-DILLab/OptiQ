"""MoveCar, matched-width baselines, and unchanged production OptiQ updates."""
import argparse,json,time
from pathlib import Path
import numpy as np
import gymnasium as gym
import jax,jax.numpy as jnp,flax.linen as nn,optax
from flax import serialization
from flax.training.train_state import TrainState
from stable_baselines3.common.logger import configure
from optiq_dime import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from .shared import config,begin


def dynamics(x,a): return np.clip(x+a,0,10)
def reward(x): return 2*((x>=.5)&(x<=1.5))+1*((x>=8.5)&(x<=9.5))

class MoveCar(gym.Env):
    metadata={}
    def __init__(self):
        self.observation_space=gym.spaces.Box(0,10,(1,),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.x=8.; self.t=0
        return np.array([self.x],np.float32),{}
    def step(self,action):
        self.x=float(dynamics(self.x,float(action[0]))); self.t+=1
        return np.array([self.x],np.float32),float(reward(self.x)),False,self.t>=100,{}

class DeterministicActor(nn.Module):
    @nn.compact
    def __call__(self,obs):
        x=obs
        for _ in range(3): x=nn.gelu(nn.Dense(256)(x))
        # Preserve the bounded smooth output of the deterministic baselines.
        return jnp.tanh(nn.Dense(1)(x))

class Critic(nn.Module):
    @nn.compact
    def __call__(self,obs,action):
        x=jnp.concatenate((obs,action),-1)
        for _ in range(3): x=nn.gelu(nn.Dense(256)(x))
        return nn.Dense(1)(x)[...,0]

class Baseline:
    def __init__(self,method,seed,importance='no_is'):
        self.method=method; self.importance=importance
        self.nq=1 if method in ('ddpg','sd2') else 2
        self.na=2 if method=='sd3' else 1
        self.actor=DeterministicActor(); self.critic=Critic()
        key=jax.random.PRNGKey(seed); keys=jax.random.split(key,5)
        self.actors=tuple(TrainState.create(apply_fn=self.actor.apply,params=self.actor.init(keys[i],jnp.zeros((1,1)))['params'],tx=optax.adam(3e-4)) for i in range(self.na))
        self.critics=tuple(TrainState.create(apply_fn=self.critic.apply,params=self.critic.init(keys[i+2],jnp.zeros((1,1)),jnp.zeros((1,1)))['params'],tx=optax.adam(3e-4)) for i in range(self.nq))
        self.at=tuple(a.params for a in self.actors); self.qt=tuple(q.params for q in self.critics)
        self.updates=0; self.key=keys[-1]
        self._update=jax.jit(self._update_impl,static_argnames=('actor_due',))
        self._act=jax.jit(self._act_impl)
        self._q=jax.jit(lambda qs,s,a:jnp.stack([self.critic.apply({'params':q.params},s,a) for q in qs]))
    def _act_impl(self,actors,critics,s):
        actions=[a.apply_fn({'params':a.params},s) for a in actors]
        if len(actions)==1: return actions[0]
        vals=jnp.stack([q.apply_fn({'params':q.params},s,a) for q,a in zip(critics,actions)])
        return jnp.where((vals[0]>=vals[1])[:,None],actions[0],actions[1])
    def act(self,s,key): return self._act(self.actors,self.critics,jnp.asarray(s))
    def q(self,s,a): return self._q(self.critics,jnp.asarray(s),jnp.asarray(a))
    def _update_impl(self,actors,critics,at,qt,batch,key,actor_due):
        s,a,r,sp=batch; keys=jax.random.split(key,4)
        targets=[]
        for i in range(self.nq):
            ac=self.actor.apply({'params':at[i if self.method=='sd3' else 0]},sp)
            if self.method in ('sd2','sd3'):
                noise=jax.random.normal(keys[i],(len(s),50,1))*.2
                candidate=jnp.clip(ac[:,None,:]+jnp.clip(noise,-.5,.5),-1,1)
                spp=jnp.broadcast_to(sp[:,None,:],candidate.shape)
                values=jnp.stack([self.critic.apply({'params':p},spp,candidate) for p in qt]).min(0)
                logits=values/.25
                if self.importance=='official_is': logits-=jnp.sum(-.5*(noise/.2)**2-jnp.log(.2*jnp.sqrt(2*jnp.pi)),axis=-1)
                value=jnp.sum(jax.nn.softmax(logits,axis=1)*values,axis=1)
            else:
                if self.method=='td3': ac=jnp.clip(ac+jnp.clip(jax.random.normal(keys[i],ac.shape)*.2,-.5,.5),-1,1)
                value=jnp.stack([self.critic.apply({'params':p},sp,ac) for p in qt]).min(0)
            targets.append(jax.lax.stop_gradient(r+.99*value))
        nextcrit=[]; losses=[]
        for q,target in zip(critics,targets):
            def loss(p): return jnp.mean((q.apply_fn({'params':p},s,a)-target)**2)
            value,grad=jax.value_and_grad(loss)(q.params); nextcrit.append(q.apply_gradients(grads=grad)); losses.append(value)
        critics=tuple(nextcrit)
        if actor_due:
            nextactors=[]
            for i,actor in enumerate(actors):
                q=critics[i if self.method=='sd3' else 0]
                def aloss(p): return -jnp.mean(q.apply_fn({'params':q.params},s,actor.apply_fn({'params':p},s)))
                nextactors.append(actor.apply_gradients(grads=jax.grad(aloss)(actor.params)))
            actors=tuple(nextactors)
            at=tuple(jax.tree.map(lambda t,p:.995*t+.005*p,t,a.params) for t,a in zip(at,actors))
            qt=tuple(jax.tree.map(lambda t,p:.995*t+.005*p,t,q.params) for t,q in zip(qt,critics))
        return actors,critics,at,qt,jnp.mean(jnp.array(losses))
    def update(self,batch):
        self.key,key=jax.random.split(self.key); self.updates+=1
        self.actors,self.critics,self.at,self.qt,loss=self._update(self.actors,self.critics,self.at,self.qt,tuple(map(jnp.asarray,batch)),key,actor_due=self.method!='td3' or self.updates%2==0)
        return loss
    def save(self,path):
        Path(path).write_bytes(serialization.to_bytes({'actors':self.actors,'critics':self.critics,'at':self.at,'qt':self.qt,'key':self.key,'updates':self.updates}))
    def load(self,path):
        d=serialization.from_bytes({'actors':self.actors,'critics':self.critics,'at':self.at,'qt':self.qt,'key':self.key,'updates':self.updates},Path(path).read_bytes())
        for k,v in d.items(): setattr(self,k,v)

class ProductionOptiQ:
    def __init__(self,seed,warmup=5000):
        import copy
        cfg=copy.deepcopy(config(seed)); cfg.diagnostic_interval=0; cfg.env_name="OptiQMoveCar-v0"; cfg.wandb.activate=False
        cfg.alg.learning_starts=warmup; cfg.alg.actor.learning_starts=warmup
        self.model=OptiQDIME('MlpPolicy',MoveCar(),model_save_path=None,save_every_n_steps=5000,cfg=cfg)
        self.model.set_logger(configure(None,[]))
        self._act=jax.jit(lambda state,s,key:OptiQPolicy.sample_action(state,s,key,False))
        self._q=jax.jit(lambda state,s,a:state.apply_fn({'params':state.params,'batch_stats':state.batch_stats},s,a,train=False)[...,0])
    def act(self,s,key): return self._act(self.model.policy.actor_state,jnp.asarray(s),key)
    def q(self,s,a): return self._q(self.model.policy.qf_state,jnp.asarray(s),jnp.asarray(a))
    def add(self,s,a,r,sp,step):
        self.model.replay_buffer.add(s[None],sp[None],a[None],np.array([r]),np.array([False]),[{}]); self.model.num_timesteps=step
    def update(self,batch=None): self.model.train(256,1)
    def save(self,path):
        p=self.model.policy
        Path(path).write_bytes(serialization.to_bytes({'actor':p.actor_state,'target_actor':p.target_actor_state,'critic':p.qf_state,'key':jax.random.key_data(self.model.key),'updates':self.model._n_updates}))
    def load(self,path):
        p=self.model.policy
        d=serialization.from_bytes({'actor':p.actor_state,'target_actor':p.target_actor_state,'critic':p.qf_state,'key':jax.random.key_data(self.model.key),'updates':self.model._n_updates},Path(path).read_bytes())
        p.actor_state=d['actor']; p.target_actor_state=d['target_actor']; p.qf_state=d['critic']; self.model.key=jax.random.wrap_key_data(d['key']); self.model._n_updates=d['updates']

def make_agent(method,seed,importance='no_is',warmup=5000):
    return ProductionOptiQ(seed,warmup) if method=='optiq' else Baseline(method,seed,importance)

def evaluate(agent,seed,episodes=10,horizon=100,starts=None):
    key=jax.random.PRNGKey(seed); x=np.full((episodes,1),8.,np.float32) if starts is None else np.asarray(starts,np.float32).copy()
    ret=np.zeros(len(x)); discounted=np.zeros(len(x)); occupancy=np.zeros(len(x))
    for t in range(horizon):
        key,ak=jax.random.split(key); a=np.asarray(agent.act(x,ak)); x=dynamics(x,a).astype('float32'); r=reward(x)[:,0]
        ret+=r; discounted+=(.99**t)*r; occupancy+=((x[:,0]>=.5)&(x[:,0]<=1.5))
    return ret,discounted,occupancy/horizon

def run(args):
    out=begin(args.out,args); agent=make_agent(args.method,args.seed,args.importance,args.warmup); env=MoveCar()
    s,_=env.reset(seed=args.seed); rng=np.random.default_rng(args.seed); key=jax.random.PRNGKey(args.seed+500)
    buffer=np.zeros((args.steps,4),np.float32); logs=[]; start=time.monotonic()
    grid=np.linspace(0,10,256,dtype=np.float32)[:,None]
    def checkpoint(t):
        agent.save(out/f'checkpoint_{t}.msgpack')
        ret,_,occ=evaluate(agent,args.seed+90000)
        pk=jax.random.PRNGKey(args.seed+90001)
        obs=np.repeat(grid,128,axis=0); actions=np.asarray(agent.act(obs,pk)).reshape(256,128)
        # Continuing value truth uses the same frozen policy, not a max-policy return.
        starts=np.linspace(0,10,32,dtype=np.float32)[:,None]
        starts=np.repeat(starts,8,axis=0)
        vk=jax.random.PRNGKey(args.seed+90002); va=np.asarray(agent.act(starts,vk))
        q=np.asarray(agent.q(starts,va)); q=q.mean(0)
        # Include the exact first action evaluated by the critic.
        xp=dynamics(starts,va).astype('float32')
        _,tail,_=evaluate(agent,args.seed+90003,horizon=args.value_horizon,starts=xp)
        true=reward(xp)[:,0]+.99*tail
        np.savez_compressed(out/f'probe_{t}.npz',states=grid,actions=actions,value_states=starts,value_actions=va,q=q,true=true,bias=q-true,return_samples=ret)
        logs.append(dict(step=t,return_mean=float(ret.mean()),good_occupancy=float(occ.mean()),q_mean=float(q.mean()),true_mean=float(true.mean()),bias=float((q-true).mean()),elapsed=time.monotonic()-start))
        (out/'learning.json').write_text(json.dumps(logs,indent=2)); print(json.dumps(logs[-1]),flush=True)
    checkpoint(0)
    for t in range(1,args.steps+1):
        key,ak=jax.random.split(key)
        a=rng.uniform(-1,1,1).astype('float32') if t<=args.warmup else np.asarray(agent.act(s[None],ak))[0]
        if args.method!='optiq' and t>args.warmup: a=np.clip(a+rng.normal(0,.5,1),-1,1).astype('float32')
        sp,r,_,trunc,_=env.step(a); buffer[t-1]=[s[0],a[0],r,sp[0]]
        if args.method=='optiq': agent.add(s,a,r,sp,t)
        if t>args.warmup:
            if args.method=='optiq': agent.update()
            else:
                b=buffer[rng.integers(t,size=256)]; agent.update((b[:,0:1],b[:,1:2],b[:,2],b[:,3:4]))
        s=env.reset()[0] if trunc else sp
        if t==args.warmup+1 or t%args.interval==0 or t==args.steps: checkpoint(t)
    np.savez_compressed(out/'replay.npz',transitions=buffer)
    (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['optiq','ddpg','sd2','td3','sd3'],required=True); ap.add_argument('--importance',choices=['no_is','official_is'],default='no_is')
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=1000000); ap.add_argument('--warmup',type=int,default=5000)
    ap.add_argument('--interval',type=int,default=5000); ap.add_argument('--value-horizon',type=int,default=2048); ap.add_argument('--out',required=True)
    run(ap.parse_args())
