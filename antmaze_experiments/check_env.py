"""Audit exact upstream sparse physics/configs and256-worker execution."""
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
from .envs import vector, transition, make_one, ROOT
from .run import write
from .settings import NUM_ENVS, UPDATES, WARMUP, BUDGETS, total_budget, expected_updates


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    from gym.utils.seeding import np_random
    rng,_=np_random(42);restored=pickle.loads(pickle.dumps(rng))
    assert np.array_equal(rng.random(1000),restored.random(1000))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from ddiffpg.utils.common import preprocess_cfg
    configs={}
    for task in BUDGETS:
        for method in ('sac','dipo'):
            with initialize_config_dir(config_dir=str(ROOT/'antmaze/ddiffpg/cfg'),version_base=None):
                cfg=compose(config_name='default',overrides=[f'algo={method}_algo',f'env.name=antmaze-{task}','seed=0'])
            cfg=preprocess_cfg(cfg,if_ddiffpg=False)
            assert cfg.num_envs==NUM_ENVS==256 and cfg.algo.update_times==UPDATES==8
            assert cfg.algo.batch_size==4096 and cfg.algo.warm_up*NUM_ENVS==WARMUP
            assert cfg.env.reward_type=='sparse' and cfg.max_step==BUDGETS[task]
            assert cfg.intrinsic.type=='noveld' and not cfg.intrinsic.normalize
            if method=='dipo':assert cfg.algo.v_min==0 and cfg.algo.v_max==5
            # Reproduce the original loop, including strict > and warmup exclusion.
            global_steps=0;updates=0
            while True:
                global_steps+=NUM_ENVS;updates+=UPDATES
                if global_steps>cfg.max_step:break
            assert global_steps+WARMUP==total_budget(task)
            assert updates==expected_updates(total_budget(task))
            configs[f'{task}-{method}']=dict(num_envs=cfg.num_envs,batch_size=cfg.algo.batch_size,
                update_times=cfg.algo.update_times,warmup=cfg.algo.warm_up,
                max_step=cfg.max_step,total_interactions=total_budget(task),updates=updates,
                random_init=cfg.env.random_init,actor_lr=cfg.algo.actor_lr,critic_lr=cfg.algo.critic_lr,
                gamma=cfg.algo.gamma,tau=cfg.algo.tau)
    results={}
    for task in BUDGETS:
        began=time.monotonic()
        single=make_one(task,0);o=single.reset();state=single.state();physics=single.physics_env
        assert o.shape==(29,) and physics.frame_skip==5 and physics.random_init==(task=='v1')
        assert np.isclose(physics.model.opt.timestep,.02)
        assert np.all(physics.model.actuator_gear[:,0]==30)
        assert np.all(single.action_space.low==-1) and np.all(single.action_space.high==1)
        assert single.env.spec.kwargs['maze_size_scaling']==4.0
        action=np.linspace(-.2,.2,8)
        wrapped=single.step(action);single.restore(state);raw=single.env.step(action)
        assert np.allclose(wrapped[0],raw[0],atol=1e-6) and wrapped[1:3]==raw[1:3]
        assert wrapped[3]['success']==raw[3]['success']
        goals=np.asarray(physics.target_goal).reshape(-1,2)
        rewards=[]
        for goal in goals:
            single.reset();physics.set_xy(goal)
            _,reward,done,info=single.step(np.zeros(8))
            assert done and info['success']>0
            assert reward==(20 if tuple(goal)==(-8,8) else 10)
            rewards.append(reward)
        single.close()
        env=vector(task,NUM_ENVS,0);obs=env.reset();assert obs.shape==(NUM_ENVS,29)
        horizon=500 if task in ('v1','v2') else 700
        for t in range(horizon):
            nxt,r,done,infos=env.step(np.zeros((NUM_ENVS,8),np.float32))
            final,terminal=transition(nxt,done,infos)
            assert np.isfinite(final).all() and np.isin(r,[0,10,20]).all()
        assert done.all() and not terminal.any()
        assert all('terminal_observation' in i and i['TimeLimit.truncated'] for i in infos)
        states=env.call('state');assert len(states)==NUM_ENVS
        env.close()
        results[task]=dict(passed=True,goals=goals.tolist(),goal_rewards=rewards,
            horizon=horizon,observations=[NUM_ENVS,29],actions=[NUM_ENVS,8],
            frame_skip=5,timestep=.02,gear=30,maze_size_scaling=4,
            random_init=task=='v1',raw_wrapper_reward_done_parity=True,seconds=time.monotonic()-began)
    import torch
    from ddiffpg.replay.simple_replay import ReplayBuffer
    for capacity in (2*NUM_ENVS,2*NUM_ENVS+88):
        memory=ReplayBuffer(capacity,(29,),8,device='cpu')
        for start in range(0,8*NUM_ENVS,NUM_ENVS):
            ids=torch.arange(start,start+NUM_ENVS,dtype=torch.float32)
            memory.add_to_buffer([ids[:,None].repeat(1,29),torch.zeros(NUM_ENVS,8),
                                  ids,ids[:,None].repeat(1,29),torch.zeros(NUM_ENVS)])
            assert memory.total_samples==start+NUM_ENVS
            assert memory.cur_capacity==min(start+NUM_ENVS,capacity)
            actual=memory.buf_obs[:memory.cur_capacity,0].sort().values
            assert torch.equal(actual,torch.arange(max(0,start+NUM_ENVS-capacity),start+NUM_ENVS,dtype=torch.float32))
    from ddiffpg.utils.intrinsic import IntrinsicM
    intrinsic=IntrinsicM((29,),env_name='antmaze-v1',normalize=False,pos_enc=True,L=10,device='cpu')
    obs=torch.randn(8,29);nxt=torch.randn(8,29)
    expected=.01*torch.clamp(intrinsic.get_novelty(intrinsic.encode_obs(nxt))-.5*intrinsic.get_novelty(intrinsic.encode_obs(obs)),min=0)
    assert torch.equal(intrinsic.compute_reward(obs,nxt).ravel(),expected)
    write(a.output,dict(passed=True,gym_rng_pickle_sequence_preserved=True,tasks=results,
        configs=configs,replay_wrap_verified=True,upstream_noveld_verified=True))


if __name__=='__main__':main()
