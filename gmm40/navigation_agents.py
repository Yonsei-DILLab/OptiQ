"""Original DIPO, MEow and MFPO online updates on a common navigation task."""
import copy
import sys
from types import SimpleNamespace
import numpy as np
from .target import ROOT


class DIPOOnline:
    def __init__(self,env,seed,folder):
        import torch
        sys.path.insert(0,str(ROOT/"gmm40-baseline/DIPO"))
        from agent.DiPo import DiPo
        from agent.replay_memory import ReplayMemory,DiffusionMemory
        class CorrectedDiffusionMemory(DiffusionMemory):
            def replace(self,indices,actions):self.best_actions[indices]=actions
        self.device=torch.device('cuda')
        torch.manual_seed(seed);np.random.seed(seed)
        self.memory=ReplayMemory(2,2,1000000,self.device)
        self.dmemory=CorrectedDiffusionMemory(2,2,1000000,self.device)
        args=SimpleNamespace(policy_type="Diffusion",noise_ratio=1.,beta_schedule="cosine",n_timesteps=100,
             diffusion_lr=3e-4,critic_lr=3e-4,action_gradient_steps=20,ratio=.1,ac_grad_norm=2.,tau=.005,
             update_actor_target_every=1,action_lr=.03)
        self.agent=DiPo(args,2,env.action_space,self.memory,self.dmemory,self.device)
        self.updates=0

    def act(self,obs):
        import torch
        with torch.no_grad():return self.agent.actor(torch.as_tensor(obs,device=self.device),eval=False).cpu().numpy()

    def q(self,obs,actions):
        import torch
        with torch.no_grad():return self.agent.critic.q_min(torch.as_tensor(obs,device=self.device),torch.as_tensor(actions,device=self.device)).cpu().numpy()

    def store(self,s,a,r,ns,done):self.agent.append_memory(s,a,r,ns,.99*(1-done))

    def update(self):
        self.agent.train(1,256);self.updates+=1
        return {"actor_updates":self.updates}

    def save(self,path):
        import torch
        torch.save(dict(actor=self.agent.actor.state_dict(),actor_target=self.agent.actor_target.state_dict(),critic=self.agent.critic.state_dict(),critic_target=self.agent.critic_target.state_dict(),actor_optimizer=self.agent.actor_optimizer.state_dict(),critic_optimizer=self.agent.critic_optimizer.state_dict(),updates=self.updates,memory=self.memory.__dict__,diffusion_memory=self.dmemory.__dict__,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)


class MEowOnline:
    def __init__(self,env,seed,folder):
        import torch
        sys.path.insert(0,str(ROOT/"gmm40-baseline/meow/toy"))
        from agents.meow import MEOW
        torch.manual_seed(seed);np.random.seed(seed)
        args=SimpleNamespace(device="cuda",save_path=str(folder/"tensorboard"),buffer_size=1000000,state_sizes=2,action_sizes=2,
                alpha=1.,sigma_max=1.,sigma_min=-2.,lr=1e-3,grad_clip=30.,gamma=.99,tau=.005,batch_size=256)
        self.agent=MEOW(args);self.updates=0

    def act(self,obs):
        import torch
        self.agent.policy.eval()
        with torch.no_grad():return self.agent.act(torch.as_tensor(obs,device="cuda"))[0].cpu().numpy()

    def q(self,obs,actions):
        import torch
        self.agent.policy.eval()
        obs=torch.as_tensor(obs,device="cuda");act=torch.as_tensor(actions,device="cuda")
        with torch.no_grad():
            q,_=self.agent.policy.get_qv(torch.cat([obs,obs]),torch.cat([act,act]))
            return torch.minimum(q[:len(obs)],q[len(obs):]).cpu().numpy()

    def store(self,s,a,r,ns,done):self.agent.buffer.store(s,a,r,ns,float(done))

    def update(self):
        info=self.agent.update();self.updates+=1
        return {k:float(v) for k,v in info.items()}

    def save(self,path):
        import torch
        torch.save(dict(policy=self.agent.policy.state_dict(),target=self.agent.policy_old.state_dict(),optimizer=self.agent.optim.state_dict(),buffer=self.agent.buffer.__dict__,updates=self.updates,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)


class MFPOOnline:
    def __init__(self,env,seed,folder):
        sys.path.append('/root/.venv-gmm40/lib/python3.11/site-packages')
        sys.path.insert(0,str(ROOT/'gmm40-baseline/MFPO'))
        import gym
        import jax
        import jax.numpy as jnp
        from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
        from jaxrl5.networks.mean_flow import action_sampler,action_sampler_with_logp
        from jaxrl5.data.replay_buffer import ReplayBuffer
        obs_space=gym.spaces.Box(-50.,50.,shape=(2,),dtype=np.float32)
        act_space=gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.agent=MeanFlowLearner.create(seed,obs_space,act_space,actor_hidden_dims=(256,256,256),critic_hidden_dims=(256,256,256),temp=.01,temp_lr=3e-4,T=2,eval_action_selection=False)
        self.buffer=ReplayBuffer(obs_space,act_space,1000000)
        self.buffer.seed(seed)
        self.updates=0
        self.eval_key=None
        # Moving Q uses the upstream adaptive temperature; fixed-Q uses a separate adapter.
        self.q_fn=jax.jit(lambda agent,s,a:agent.calc_value(s,a))
        self.act_fn=jax.jit(lambda agent,key,obs:action_sampler(agent.actor.apply_fn,agent.actor.params,agent.T,jax.random.normal(key,(len(obs),2)),obs,agent.clip_sampler))
        @jax.jit
        def support_probe(agent,batch):
            # Independent deterministic probe; never split the learner's RNG.
            noise=jax.random.normal(jax.random.PRNGKey(92137),(batch['actions'].shape[0],2))
            next_actions,logp=action_sampler_with_logp(agent.actor.apply_fn,agent.actor.params,
                    agent.logp_mvel.apply_fn,agent.logp_mvel.params,agent.T,noise,batch['next_observations'],agent.clip_sampler)
            temperature=agent.temp.apply_fn({'params':agent.temp.params})
            atoms=agent.z_atoms[None,:]-temperature*logp[:,None] if agent.backup_entropy else agent.z_atoms[None,:]
            raw_target=batch['rewards'][:,None]+agent.discount*batch['masks'][:,None]*atoms
            clipped=jnp.clip(raw_target,agent.v_min,agent.v_max)
            target_probs=[];current_probs=[]
            for critic,target in ((agent.critic_1,agent.target_critic_1),(agent.critic_2,agent.target_critic_2)):
                target_probs.append(jax.nn.softmax(target.apply_fn({'params':target.params},batch['next_observations'],next_actions),axis=-1))
                current_probs.append(jax.nn.softmax(critic.apply_fn({'params':critic.params},batch['observations'],batch['actions']),axis=-1))
            p=jnp.stack(target_probs).mean(0);current=jnp.stack(current_probs).mean(0)
            return {'value_support/target_clipped_below_mass':(p*(raw_target<agent.v_min)).sum(-1).mean(),
                    'value_support/target_clipped_above_mass':(p*(raw_target>agent.v_max)).sum(-1).mean(),
                    'value_support/mean_clipping_shift':(p*(clipped-raw_target)).sum(-1).mean(),
                    'value_support/raw_expected_target_min':(p*raw_target).sum(-1).min(),
                    'value_support/current_lower_endpoint_mass':current[:,0].mean(),
                    'value_support/current_upper_endpoint_mass':current[:,-1].mean()}
        self.support_probe_fn=support_probe

    def act(self,obs):
        import jax
        if self.eval_key is None:
            key,rng=jax.random.split(self.agent.rng);self.agent=self.agent.replace(rng=rng)
        else:key,self.eval_key=jax.random.split(self.eval_key)
        return np.asarray(self.act_fn(self.agent,key,obs))

    def q(self,obs,actions):return np.asarray(self.q_fn(self.agent,obs,actions))

    def store(self,s,a,r,ns,done):
        self.buffer.insert(dict(observations=s,actions=a,rewards=np.float32(r),masks=np.float32(1-done),dones=bool(done),next_observations=ns))

    def update(self):
        batch=self.buffer.sample(256)
        self.agent,info=self.agent.update(batch);self.updates+=1
        if self.updates%1000==0:
            info={**info,**self.support_probe_fn(self.agent,batch)}
        return {k:float(v) for k,v in info.items()}

    def save(self,path):
        import flax.serialization
        import pickle
        path.write_bytes(flax.serialization.to_bytes(self.agent))
        with path.with_suffix('.replay.pkl').open('wb') as f:pickle.dump(self.buffer,f)
