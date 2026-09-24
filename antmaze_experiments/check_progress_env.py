"""CPU MuJoCo checks for new reward profiles; no learner or experiment launch."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .progress_reward import PROFILES,distance,specification,bonus_enabled


def check():
    from .envs import make_one,vector,transition
    results={}
    for task in ('v1','v2','v3','v4'):
        envs={p:make_one(task,123,reward_profile=p) for p in ('sparse','dense')+PROFILES}
        try:
            obs={p:e.reset() for p,e in envs.items()}
            for o in obs.values():np.testing.assert_array_equal(o,obs['sparse'])
            rng=np.random.default_rng(456);max_error=0.
            for step in range(100):
                action=rng.uniform(-1,1,8);rows={}
                for profile,e in envs.items():
                    before=e.physics_env.get_xy().copy()
                    o,r,d,i=e.step(action);rows[profile]=(o,r,d,i)
                    if profile=='dense':assert np.isclose(r,-i['distance'])
                    if profile in PROFILES:
                        expected=distance(before,task,profile)-distance(e.physics_env.get_xy(),task,profile)-.01+(i['upstream_sparse_reward'] if bonus_enabled(profile) else 0)
                        max_error=max(max_error,abs(float(r-expected)))
                        assert np.isclose(r,expected,atol=1e-9)
                        assert np.isclose(r,i['reward_progress']+i['reward_step_penalty']+i['reward_success'])
                for profile,(o,r,d,i) in rows.items():
                    np.testing.assert_array_equal(o,rows['sparse'][0]);assert d==rows['sparse'][2]
                if rows['sparse'][2]:
                    for e in envs.values():e.reset()
            arrivals=[]
            for profile in PROFILES:
                e=envs[profile];goals=np.asarray(e.physics_env.target_goal).reshape(-1,2)
                for goal in goals:
                    e.reset();state=e.state();state['qpos'][:2]=goal;state['qvel'][:]=0
                    e.restore(state);saved=e.state();a=np.zeros(8)
                    one=e.step(a);e.restore(saved);two=e.step(a)
                    np.testing.assert_allclose(one[0],two[0],atol=1e-7,rtol=1e-7)
                    assert np.isclose(one[1],two[1],atol=1e-9) and one[2]==two[2]
                    assert one[2] and one[3]['success']>0
                    wanted=(20 if tuple(goal)==(-8,8) else 10) if bonus_enabled(profile) else 0
                    assert one[3]['reward_success']==wanted
                    arrivals.append(dict(profile=profile,goal=goal.tolist(),bonus=wanted,reward=float(one[1])))
            results[task]=dict(transitions_per_profile=100,physics_identical=True,max_reward_error=max_error,
                               success_terminates_and_restores=True,arrivals=arrivals)
        finally:
            for e in envs.values():e.close()
        # Gym auto-reset must not substitute new start XY into terminal reward/replay.
        for profile in PROFILES:
            v=vector(task,2,71,asynchronous=False,reward_profile=profile)
            try:
                v.reset()
                for e in v.envs:
                    state=e.state();state['qpos'][:2]=np.asarray(e.physics_env.target_goal).reshape(-1,2)[0];state['qvel'][:]=0;e.restore(state)
                nxt,r,d,infos=v.step(np.zeros((2,8)))
                end,terminal=transition(nxt,d,infos)
                assert d.all() and terminal.all()
                for i,info in enumerate(infos):
                    np.testing.assert_array_equal(end[i],info['terminal_observation'])
                    assert np.isclose(r[i],info['reward_progress']-.01+info['reward_success'])
                # Force a time limit without arrival; bootstrap remains enabled.
                for e in v.envs:
                    state=e.state();state['elapsed']=v.envs[0].env._max_episode_steps-1;e.restore(state)
                nxt,r,d,infos=v.step(np.zeros((2,8)));end,terminal=transition(nxt,d,infos)
                assert d.all() and not terminal.any()
                assert all(i['reward_success']==0 for i in infos)
            finally:v.close()
        results[task]['autoreset_and_timeout_verified']=True
    # Forked vector path, including shared immutable distance precomputation.
    for task in ('v1','v2','v3','v4'):
        v=vector(task,4,77,asynchronous=True,reward_profile=PROFILES[1])
        try:
            v.reset();out=v.step(np.zeros((4,8)));assert np.isfinite(out[1]).all()
        finally:v.close()
    return dict(verified=True,training_launched=False,results=results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    out=check()
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
