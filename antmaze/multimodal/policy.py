"""Learned-policy draws only: no best-of-K or external behavior noise."""
from contextlib import contextmanager
import numpy as np
from antmaze.agents import SB3,DIPO,MEOW,MFPO,SQL


class PolicyView:
    def __init__(self,agent):self.agent=agent;self.mode="policy"
    @contextmanager
    def evaluation(self,mode,seed):
        self.mode=mode
        with self.agent.evaluation("stochastic",seed):yield
    def act(self,obs):
        a=self.agent
        if isinstance(a,SB3):
            if a.method=="optiq":
                p=a.model.policy;p.reset_noise()
                return np.asarray(p.sample_action(p.actor_state,obs,p.noise_key,
                    deterministic=False,sample_conditional_noise=self.mode=="policy"))
            return a.model.predict(obs,deterministic=False)[0]
        if isinstance(a,MFPO):
            actions,a.eval_agent=a.eval_agent.eval_actions_sample_batch(obs)
            return np.asarray(actions)
        if isinstance(a,SQL):return a.act(obs)
        import torch
        with torch.no_grad():
            if isinstance(a,DIPO):return a.agent.actor(torch.as_tensor(obs,device="cuda"),eval=False).cpu().numpy()
            a.policy.eval()
            return a.policy.sample(len(obs),obs,deterministic=False)[0].cpu().numpy()
