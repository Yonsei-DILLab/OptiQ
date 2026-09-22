"""Runtime physics and vector API audit, without training or reward changes."""
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
            assert done and info['success']>0 and reward in (10,20)
            rewards.append(reward)
        single.close()
        env=vector(task,64,0);obs=env.reset();assert obs.shape==(64,29)
        expected=500 if task in ('v1','v2') else 700
        for t in range(expected):
            nxt,r,done,infos=env.step(np.zeros((64,8),np.float32))
            final,terminal=transition(nxt,done,infos)
            assert np.isfinite(final).all() and set(r)<=set([0,10,20])
        assert done.all() and not terminal.any()
        assert all('terminal_observation' in i and i['TimeLimit.truncated'] for i in infos)
        states=env.call('state');assert len(states)==64
        env.close()
        results[task]=dict(passed=True,goals=goals.tolist(),goal_rewards=rewards,
            horizon=expected,observations=[64,29],actions=[64,8],seconds=time.monotonic()-began)
    write(a.output,dict(passed=True,gym_rng_pickle_sequence_preserved=True,tasks=results))


if __name__=='__main__':main()
