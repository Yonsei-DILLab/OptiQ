"""Fixed-Q adapters for SAC and the supplied DIPO / MEow implementations."""
import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .target import ROOT


def log_jacobian(u):
    return (math.log(40)+2*(math.log(2)-u-F.softplus(-2*u))).sum(-1)


class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.mu=nn.Linear(256,2); self.ls=nn.Linear(256,2)
        nn.init.zeros_(self.mu.bias); nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias,math.log(.5))

    def forward(self,obs,noise=None):
        h=self.net(obs); mu=self.mu(h); ls=self.ls(h).clamp(-5,2)
        if noise is None: noise=torch.randn_like(mu)
        u=mu+ls.exp()*noise
        logp=(-.5*noise.square()-ls-.5*math.log(2*math.pi)).sum(-1)-log_jacobian(u)
        return u,logp,mu,ls


class Base:
    def __init__(self,target,seed,batch):
        torch.set_num_threads(1)
        assert torch.cuda.is_available(),"GPU required"
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.device=torch.device("cuda")
        self.target,self.batch,self.updates,self.queries=target,batch,0,0

    def advance(self,count):
        info={}
        for _ in range(count):
            info=self.update(); self.updates+=1
        torch.cuda.synchronize()
        return {**{k:float(v) for k,v in info.items()},"Q_evaluations":self.queries}

    def save(self,path):
        state=dict(actor=self.actor.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,
                   queries=self.queries,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if hasattr(self,"buffer"):
            state.update(buffer=self.buffer,buffer_size=self.buffer_size,buffer_position=self.buffer_position)
        torch.save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location=self.device,weights_only=False)
        self.actor.load_state_dict(state["actor"]); self.optimizer.load_state_dict(state["optimizer"])
        self.updates,self.queries=state["updates"],state["queries"]
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        if "buffer" in state:
            self.buffer,self.buffer_size,self.buffer_position=state["buffer"],state["buffer_size"],state["buffer_position"]

    def initialize_buffer(self,pretanh=False):
        self.buffer=torch.empty((1000000,2),device=self.device)
        initial=torch.rand((5000,2),device=self.device)*1.9998-.9999
        self.buffer[:5000]=torch.atanh(initial) if pretanh else initial
        self.buffer_size=self.buffer_position=5000

    def append(self,values):
        indices=(torch.arange(len(values),device=self.device)+self.buffer_position)%len(self.buffer)
        self.buffer[indices]=values.detach()
        self.buffer_position=(self.buffer_position+len(values))%len(self.buffer)
        self.buffer_size=min(len(self.buffer),self.buffer_size+len(values))


class SAC(Base):
    def __init__(self,target,seed,batch):
        super().__init__(target,seed,batch)
        self.actor=GaussianActor().to(self.device)
        self.optimizer=torch.optim.Adam(self.actor.parameters(),lr=3e-4)

    def update(self):
        u,logp,mu,ls=self.actor(torch.zeros((self.batch,1),device=self.device))
        q=self.target.torch_log_prob(40*torch.tanh(u))
        loss=(logp-q).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
        self.queries+=self.batch
        return dict(loss=loss.detach(),entropy=-logp.mean().detach(),Q=q.mean().detach(),sigma=ls.exp().mean().detach())

    def evaluate_samples(self,n,seed):
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            eps=torch.randn((n,2),device=self.device)
            u,_,_,_=self.actor(torch.zeros((n,1),device=self.device),eps)
            x=40*torch.tanh(u)
            rollout=torch.stack([40*torch.tanh(eps[:128]),x[:128]])
            return x.cpu().numpy(),rollout.cpu().numpy(),{}


class DIPO(Base):
    def __init__(self,target,seed,batch):
        super().__init__(target,seed,batch)
        sys.path.insert(0,str(ROOT/"gmm40-baseline/DIPO"))
        from agent.diffusion import Diffusion
        self.actor=Diffusion(1,2,1.0,beta_schedule="cosine",n_timesteps=100).to(self.device)
        self.optimizer=torch.optim.Adam(self.actor.parameters(),lr=3e-4,eps=1e-5)
        self.initialize_buffer()

    def update(self):
        # Batch 32 consecutive draws to amortize 100-step diffusion generation.
        if self.updates and self.updates%32==0:
            with torch.no_grad(): self.append(self.actor.sample(torch.zeros((32,1),device=self.device)))
        indices=torch.randint(self.buffer_size,(self.batch,),device=self.device)
        actions=self.buffer[indices].detach().clone().requires_grad_(True)
        optim=torch.optim.Adam([actions],lr=.03,eps=1e-5)
        for _ in range(20):
            q=self.target.torch_log_prob(40*actions)
            optim.zero_grad(set_to_none=True)
            (-q).sum().backward()
            nn.utils.clip_grad_norm_([actions],.2)
            optim.step()
            with torch.no_grad(): actions.clamp_(-1,1)
        self.buffer[indices]=actions.detach()
        loss=self.actor.loss(actions.detach(),torch.zeros((self.batch,1),device=self.device))
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),2.)
        self.optimizer.step()
        self.queries+=20*self.batch
        return dict(loss=loss.detach(),improved_Q=q.mean().detach(),buffer_size=self.buffer_size,action_abs_max=actions.abs().max().detach())

    def evaluate_samples(self,n,seed):
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            self.actor.noise_ratio=1.0
            u=torch.randn((n,2),device=self.device)
            obs=torch.zeros((n,1),device=self.device)
            path=[(40*u[:128]).cpu().numpy()]
            for t in reversed(range(self.actor.n_timesteps)):
                u=self.actor.p_sample(u,torch.full((n,),t,device=self.device,dtype=torch.long),obs)
                path.append((40*u[:128]).cpu().numpy())
            u=u.clamp(-1,1)
            path[-1]=(40*u[:128]).cpu().numpy()
            return (40*u).cpu().numpy(),np.stack(path),{}


class MEow(Base):
    def __init__(self,target,seed,batch):
        super().__init__(target,seed,batch)
        sys.path.insert(0,str(ROOT/"gmm40-baseline/meow/toy"))
        from modules.policy import FlowPolicy
        self.actor=FlowPolicy(alpha=1.,sigma_max=1.,sigma_min=-2.,action_sizes=2,state_sizes=1,device=self.device)
        self.optimizer=torch.optim.Adam(self.actor.parameters(),lr=1e-3)
        self.initialize_buffer()

    def update(self):
        self.actor.eval()
        with torch.no_grad():
            new,_=self.actor.sample(1,torch.zeros((1,1),device=self.device))
            self.append(new)
        indices=torch.randint(self.buffer_size,(self.batch,),device=self.device)
        a=self.buffer[indices]
        q=self.target.torch_log_prob(40*a).detach()
        obs=torch.zeros((2*self.batch,1),device=self.device)
        self.actor.train()
        qhat,_=self.actor.get_qv(obs,torch.cat([a,a],dim=0))
        # Qhat_x = Qhat_normalized - log(40^2); the learned shifts absorb constants.
        loss=((qhat[:,0]-2*math.log(40)-q.repeat(2))**2).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),30.)
        self.optimizer.step()
        self.queries+=self.batch
        return dict(loss=loss.detach(),target_Q=q.mean(),buffer_size=self.buffer_size)

    def evaluate_samples(self,n,seed):
        self.actor.eval()
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            obs=torch.zeros((n,1),device=self.device)
            u,_=self.actor.prior.sample(n,context=obs)
            path=[(40*torch.tanh(u[:128])).cpu().numpy()]
            for i,flow in enumerate(self.actor.flows):
                u,_=flow.forward(u,context=obs)
                position=40*u if i==len(self.actor.flows)-1 else 40*torch.tanh(u)
                path.append(position[:128].cpu().numpy())
            return (40*u).cpu().numpy(),np.stack(path),{}


def make_agent(method,target,seed,batch):
    return {"sac":SAC,"dipo":DIPO,"meow":MEow}[method](target,seed,batch)
