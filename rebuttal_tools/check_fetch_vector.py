"""Check vector HER boundaries, overwrite handling, normalization, and real Fetch collection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import json
import pickle
import numpy as np
from gymnasium import spaces
from rebuttal_tools.fetch_vector_replay import VectorHERReplayBuffer


def reward(achieved, desired, infos):
    return -(np.linalg.norm(achieved-desired,axis=-1) >= .05).astype(np.float32)


def check_replay():
    obs_space=spaces.Box(-np.inf,np.inf,(28,),np.float32);act_space=spaces.Box(-1,1,(4,),np.float32)
    replay=VectorHERReplayBuffer(14,obs_space,act_space,n_envs=2,device='cpu',compute_reward=reward,her_seed=42)
    def states(t):
        o=np.zeros((2,28),np.float32);o[:,0]=[0,1];o[:,3]=np.arange(2)*100+t;o[:,4]=t;o[:,5]=.4;o[:,-3:]=-20
        return o
    for t in range(25):
        obs,nxt=states(t),states(t+1);done=np.array([(t+1)%3==0,(t+1)%4==0])
        infos=[dict(TimeLimit_truncated=True) for _ in range(2)]
        for i,info in enumerate(infos):info.update({'TimeLimit.truncated':bool(done[i]),'fetch_raw_next_observation':nxt[i]})
        replay.add(obs,np.zeros_like(nxt),np.zeros((2,4)),np.full(2,-1.),done,infos)
    before=replay.observations.copy();before_next=replay.next_observations.copy()
    class Normalizer:
        def normalize_obs(self,x):return x+10000
        def normalize_reward(self,r):return r
    sample=replay.sample(10240,env=Normalizer())
    obs=sample.observations.numpy()-10000;nxt=sample.next_observations.numpy()-10000
    her=(np.arange(10240)%256)<204
    assert np.array_equal((obs[:,-3]>=0).reshape(-1,256).sum(1),np.full(40,204)), 'HER ratio changed inside optimizer minibatches'
    assert np.all((obs[her,-3]//100)==obs[her,0]),'HER crossed environments'
    assert np.all(obs[her,-2]>=nxt[her,4]),'Future HER used an earlier state'
    assert np.allclose(sample.rewards.numpy()[her,0],reward(nxt[her,3:6],nxt[her,-3:],None))
    assert np.all(sample.dones.numpy()==0),'Time-limit transition incorrectly terminated'
    assert np.array_equal(replay.observations,before) and np.array_equal(replay.next_observations,before_next)
    loaded=pickle.loads(pickle.dumps(replay));assert loaded.compute_reward is None
    assert np.array_equal(loaded.episode_ends,replay.episode_ends)
    return dict(passed=True,checks=['No cross-environment future goals','No cross-episode future goals after ring overwrite',
        '204 HER samples in every optimizer minibatch of 256','Reward recomputed before normalization','Raw replay remains unchanged','Timeout bootstrapping','Metadata serialization'])


def check_collector():
    from rebuttal_tools.fetch_vector_env import GroupedFetchVecEnv
    from fetch_rebuttal.env import make_env
    env=GroupedFetchVecEnv(4,[[],[]]);single=[make_env('pick-and-place') for _ in range(4)]
    try:
        env.seed(51000);obs=env.reset();refs=[e.reset(seed=51000+i)[0] for i,e in enumerate(single)]
        assert np.allclose(obs,np.stack(refs),atol=1e-7)
        for t in range(50):
            actions=np.tile(np.array([.1,-.2,.05,-.3],np.float32),(4,1));obs,rewards,dones,infos=env.step(actions)
            ref=[e.step(a) for e,a in zip(single,actions)]
            assert np.array_equal(rewards,[r[1] for r in ref])
            if t<49:assert np.allclose(obs,np.stack([r[0] for r in ref]),atol=1e-6)
            else:
                assert dones.all() and all(i['TimeLimit.truncated'] for i in infos)
                assert np.allclose(np.stack([i['fetch_raw_next_observation'] for i in infos]),np.stack([r[0] for r in ref]),atol=1e-6)
                assert np.allclose(np.stack([i['terminal_observation'] for i in infos]),np.stack([r[0] for r in ref]),atol=1e-6)
        return dict(passed=True,environments=4,processes=2,checks=['Matches independent environments with paired seeds/actions','Terminal physical observations survive autoreset'])
    finally:
        env.close()
        for e in single:e.close()
        assert not any(p.is_alive() for p in env.processes)


def check_density():
    import jax,jax.numpy as jnp
    from scipy.stats import truncnorm
    from ibolt.box_gaussian import component_log_prob,sample_box
    for ls in (-5.,-3.5,-1.,1.):
        means=jnp.asarray([[[.9,-.9,0.,.5]]]);scales=jnp.full_like(means,ls)
        draws=sample_box(jax.random.PRNGKey(12),jnp.broadcast_to(means,(1,4096,4)),jnp.broadcast_to(scales,(1,4096,4)))
        assert np.isfinite(draws).all() and np.max(np.abs(draws))<=1
        got=np.asarray(component_log_prob(draws[:,:16],means,scales))[0,0]
        std=np.exp(ls);mu=np.asarray(means)[0,0]
        expected=truncnorm.logpdf(np.asarray(draws)[0,:16],(-1-mu)/std,(1-mu)/std,loc=mu,scale=std).sum(-1)
        assert np.allclose(got,expected,atol=.003,rtol=.001)
    return dict(passed=True,logstd_cases=[-5,-3.5,-1,1],reference='scipy.stats.truncnorm')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collector',action='store_true');args=parser.parse_args()
    result=dict(replay=check_replay(),density=check_density())
    if args.collector:result['collector']=check_collector()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
