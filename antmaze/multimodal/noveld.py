"""DDiffPG 7edd06c NovelD port, with replay-time rewards and independent state.

Source: ddiffpg/utils/intrinsic.py and ddiffpg/models/mlp.py (Apache-2.0).
The original license is preserved in vendor/LICENSE-APACHE2.
"""
import numpy as np
import torch
from torch import nn


class NovelD:
    def __init__(self, observation_dim, seed):
        # RND initialization must not displace the policy initialization RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed + 93107)
            def net():
                return nn.Sequential(nn.Linear(observation_dim+40,512),nn.ELU(),
                    nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128))
            self.predictor=net().cuda();self.target=net().cuda()
            for model in (self.predictor,self.target):
                for layer in model.modules():
                    if isinstance(layer,nn.Linear):
                        nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
            self.target.requires_grad_(False)
        self.optimizer=torch.optim.AdamW(self.predictor.parameters(),lr=1e-4)
        self.updates=0;self.metrics={}
        self.config=dict(type="noveld",coefficient=.01,novelty_discount=.5,normalize=False,
            positional_encoding=True,frequency_bands=10,learning_rate=1e-4,gradient_clip=1.,
            upstream_commit="7edd06c4799abbab0f8fa534c21deb56253b018e")

    def encode(self,obs):
        # Exact upstream order: xy, sin(xy), cos(xy) for each frequency, rest.
        xy=obs[:,:2]
        return torch.cat([xy]+[f(xy*(2.**i)) for i in range(10)
            for f in (torch.sin,torch.cos)]+[obs[:,2:]],dim=1)

    def reward_and_update(self,obs,next_obs,env_reward):
        s=torch.as_tensor(obs,device="cuda",dtype=torch.float32)
        ns=torch.as_tensor(next_obs,device="cuda",dtype=torch.float32)
        encoded=self.encode(torch.cat([s,ns]))
        prediction=self.predictor(encoded)
        with torch.no_grad():
            target=self.target(encoded)
            novelty=torch.linalg.vector_norm(prediction-target,dim=1)
            n,nnxt=novelty.chunk(2)
            bonus=.01*torch.clamp(nnxt-.5*n,min=0)
        loss=torch.nn.functional.mse_loss(prediction,target)
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.predictor.parameters(),1.)
        self.optimizer.step();self.updates+=1
        b=bonus.cpu().numpy()
        self.metrics=dict(intrinsic_mean=float(b.mean()),intrinsic_max=float(b.max()),
            env_reward_mean=float(np.mean(env_reward)),rnd_loss=float(loss.detach()),rnd_grad=float(grad),
            rnd_updates=self.updates)
        if not all(np.isfinite(v) for v in self.metrics.values()):raise FloatingPointError(self.metrics)
        return b

    def save(self,path):
        torch.save(dict(predictor=self.predictor.state_dict(),target=self.target.state_dict(),
            optimizer=self.optimizer.state_dict(),updates=self.updates,config=self.config),path)


class Replay:
    """Uniform transition replay, re-computing novelty once per learner update."""
    def __init__(self,capacity,seed,intrinsic,device="cpu",dictionary=False):
        self.capacity=capacity;self.size=0;self.position=0
        self.rng=np.random.default_rng(seed+1729);self.intrinsic=intrinsic
        self.device=device;self.dictionary=dictionary;self.diagnostic=False
        self.data={k:np.empty((capacity,*shape),np.float32) for k,shape in
            dict(observations=(29,),actions=(8,),next_observations=(29,),rewards=(),dones=()).items()}

    def add_batch(self,obs,actions,rewards,next_obs,terminals):
        n=len(obs);indices=(self.position+np.arange(n))%self.capacity
        for key,value in zip(self.data,(obs,actions,next_obs,rewards,terminals)):
            self.data[key][indices]=value
        self.position=(self.position+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def sample(self,batch_size,env=None):
        diagnostic=self.diagnostic;self.diagnostic=False
        if diagnostic:batch_size=min(batch_size,256)  # retain the existing DACER diagnostic state count
        ix=self.rng.integers(self.size,size=batch_size)
        d={k:v[ix] for k,v in self.data.items()}
        if not diagnostic:
            d["rewards"]+=self.intrinsic.reward_and_update(d["observations"],d["next_observations"],d["rewards"])
        if self.dictionary:
            d["masks"]=1-d["dones"];return d
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        return ReplayBufferSamples(*(torch.as_tensor(d[k][:,None] if k in ("rewards","dones") else d[k],
            device=self.device) for k in ("observations","actions","next_observations","dones","rewards")))
