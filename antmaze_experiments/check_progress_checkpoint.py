"""Exercise runner checkpoint reward audit with CPU-only synthetic replay."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .progress_reward import PROFILES,maze_geometry,progress_reward,success_bonus
from .run import checkpoint


def check():
    proofs=[]
    for task in ('v1','v2','v3','v4'):
        for profile in PROFILES:
            goals=maze_geometry(task)[1]
            # Ordinary, success, and float32-roundoff radius boundary transitions.
            after=np.concatenate([np.array([[0.,0.],[.1,0.]]),goals,
                goals[:1]+[.50000001,0],goals[:1]+[.49999999,0]])
            before=after+np.array([.05,0])
            bonus=success_bonus(after,task)
            rewards,_,_=progress_reward(before,after,task,profile,bonus)
            n=len(after);obs=np.zeros((n,29),np.float32);nxt=obs.copy();obs[:,:2]=before;nxt[:,:2]=after
            replay=SimpleNamespace(buf_obs=torch.tensor(obs),buf_next_obs=torch.tensor(nxt),
                buf_action=torch.zeros((n,8)),buf_reward=torch.tensor(rewards[:,None],dtype=torch.float32),
                buf_done=torch.tensor((bonus>0)[:,None],dtype=torch.float32),
                capacity=n,cur_capacity=n,total_samples=n,next_p=0,if_full=True)
            learner=SimpleNamespace(replay=replay,intrinsic=None,noveld_enabled=False,
                updates=0,method='sac',state=lambda:dict(cpu_test=True))
            env=SimpleNamespace(call=lambda method:[dict(cpu_test=True)])
            with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.get_rng_state_all',return_value=[]):
                proof=checkpoint(learner,env,nxt[:1],Path(tmp),n,np.random.default_rng(0),
                                 dict(task=task,reward_profile=profile))
                assert proof['environment_reward_verified'] and proof['progress_replay_verified']
                assert proof['threshold_roundoff_rows']==2
                proofs.append(dict(task=task,profile=profile,replay_rows=n,boundary_rows=2,verified=True))
    assert not torch.cuda.is_initialized(), 'CPU verification must not claim a training GPU'
    return dict(verified=True,training_launched=False,proofs=proofs)

if __name__=='__main__':print(json.dumps(check(),indent=2))
