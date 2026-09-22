"""Audit original physics, dense reward, terminal observations and replay wrap."""
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
from .envs import vector, transition, make_one
from .run import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    from gym.utils.seeding import np_random
    rng,_=np_random(42);restored=pickle.loads(pickle.dumps(rng))
    assert np.array_equal(rng.random(1000),restored.random(1000))
    results={}
    for task in ('v1','v2','v3','v4'):
        began=time.monotonic()
        single=make_one(task,0);o=single.reset();state=single.state()
        action=np.linspace(-.2,.2,8)
        s1=single.step(action);single.restore(state);s2=single.step(action)
        assert np.array_equal(s1[0],s2[0]) and s1[1:3]==s2[1:3]
        goals=single.physics_env.target_goal
        goals=np.asarray(goals).reshape(-1,2)
        rewards=[]
        for goal in goals:
            single.reset();single.physics_env.set_xy(goal)
            _,reward,done,info=single.step(np.zeros(8))
            assert done and info['success']>0 and info['upstream_sparse_reward'] in (10,20)
            assert -.5 <= reward <= 0 and np.isclose(reward,-info['distance'])
            rewards.append(reward)
        single.close()
        env=vector(task,64,0);obs=env.reset();assert obs.shape==(64,29)
        expected=500 if task in ('v1','v2') else 700
        for t in range(expected):
            nxt,r,done,infos=env.step(np.zeros((64,8),np.float32))
            final,terminal=transition(nxt,done,infos)
            distance=np.linalg.norm(final[:,:2,None]-goals.T[None,:,:],axis=1).min(axis=1)
            assert np.isfinite(final).all() and (r<=0).all()
            assert np.allclose(r,-distance,rtol=2e-6,atol=2e-5)
        assert done.all() and not terminal.any()
        assert all('terminal_observation' in i and i['TimeLimit.truncated'] for i in infos)
        states=env.call('state');assert len(states)==64
        env.close()
        results[task]=dict(passed=True,goals=goals.tolist(),goal_rewards=rewards,
            horizon=expected,observations=[64,29],actions=[64,8],seconds=time.monotonic()-began)
    # With budgets above replay capacity, keep precisely the latest transitions.
    import torch
    from ddiffpg.replay.simple_replay import ReplayBuffer
    memory=ReplayBuffer(128,(29,),8,device='cpu')
    for start in range(0,512,64):
        ids=torch.arange(start,start+64,dtype=torch.float32)
        memory.add_to_buffer([ids[:,None].repeat(1,29),torch.zeros(64,8),
                              -ids,ids[:,None].repeat(1,29),torch.zeros(64)])
        assert memory.total_samples==start+64
        assert memory.cur_capacity==min(start+64,128)
        actual=memory.buf_obs[:memory.cur_capacity,0].sort().values
        assert torch.equal(actual,torch.arange(max(0,start+64-128),start+64,dtype=torch.float32))
    from .settings import DIPO_DENSE_V_MIN
    from ddiffpg.utils.distl_util import projection
    projected={}
    for lower in (0.,DIPO_DENSE_V_MIN):
        support=torch.linspace(lower,5.,51)
        mass=torch.full((2,51),1/51)
        target=projection(mass,torch.tensor([[-10.],[-30.]]),torch.ones(2,1),
                          .99,lower,5.,51,support,'cpu')
        assert torch.isfinite(target).all()
        mean=(target*support).sum(-1)
        projected[str(lower)]=mean.tolist()
        if lower==0: assert (mean==0).all()
        else: assert torch.allclose(mean,torch.tensor([-10.,-30.]),atol=.001)
    write(a.output,dict(passed=True,gym_rng_pickle_sequence_preserved=True,tasks=results,
        replay_wrap_verified=True,negative_dipo_projection=projected))


if __name__=='__main__':main()
