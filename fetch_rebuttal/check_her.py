"""HER episode, reward, normalization, ring-buffer and native-interface checks."""
import copy
import json
import pickle
from pathlib import Path
import tempfile
import numpy as np
from .her import HERReplay, SB3HERReplayBuffer, IDACHERReplay, reward_function
from .normalization import make_normalized_env


def main():
    results=[]
    for task,dim in [('pick-and-place',28),('push',28),('slide',28),('reach',13)]:
        env=make_normalized_env(task);fn=reward_function(env)
        replay=HERReplay(capacity=75,observation_dim=dim,action_dim=4,seed=7,
                         task=task,compute_reward=fn)
        sb3=SB3HERReplayBuffer(75,env.observation_space,env.action_space,device='cpu',
            n_envs=1,task=task,compute_reward=fn,her_seed=7)
        goal_slice=slice(0,3) if task=='reach' else slice(3,6)
        for counter in range(137):
            episode=counter//10
            x=np.zeros((1,dim),np.float32);y=x.copy()
            x[:,goal_slice]=np.array([episode*10+counter*.001,.3,.4],np.float32)
            y[:,goal_slice]=np.array([episode*10+(counter+1)*.001,.3,.4],np.float32)
            x[:,-3:]=y[:,-3:]=np.array([200+episode,20,20],np.float32)
            a=np.zeros((1,4),np.float32);r=fn(y[:,goal_slice],y[:,-3:],{})
            done=np.array([(counter+1)%10==0]);terminal=np.zeros(1,np.float32)
            replay.add(x,a,r,y,terminal,episode_ends=done)
            sb3.add(x,y,a,r,done,[{'TimeLimit.truncated':bool(done[0]),'fetch_raw_next_observation':y[0].copy()}])
        assert replay.her.total==137 and replay.her.completed_until==130
        assert replay.size==75 and replay.position==62
        original={k:v.copy() for k,v in replay.data.items()}
        captured={};draw=replay.her.draw
        def capture(batch):
            out=draw(batch);captured['out']=out;return out
        replay.her.draw=capture
        rng=copy.deepcopy(replay.her.rng.bit_generator.state)
        raw=replay.batch(256);indices,fidx,counters,future=captured['out']
        assert np.all((counters>=62)&(counters<130))
        assert np.all(future>=counters[:204])
        assert np.all(future//10==counters[:204]//10)
        np.testing.assert_array_equal(raw['observations'][:204,-3:],original['next_observations'][fidx,goal_slice])
        np.testing.assert_array_equal(raw['next_observations'][:204,-3:],raw['observations'][:204,-3:])
        np.testing.assert_array_equal(raw['observations'][204:],original['observations'][indices[204:]])
        np.testing.assert_array_equal(raw['rewards'],fn(raw['next_observations'][:,goal_slice],raw['next_observations'][:,-3:],{}))
        assert np.all(raw['rewards'][:204]==0) and np.all(raw['rewards'][204:]==-1)
        assert not raw['terminals'].any()
        for key in original:np.testing.assert_array_equal(replay.data[key],original[key])
        env.obs_rms.update(np.concatenate((original['observations'],original['next_observations'])))
        count=float(env.obs_rms.count)
        replay.normalizer=env;replay.her.rng.bit_generator.state=copy.deepcopy(rng)
        batch=replay.sample(256)
        sb3_batch=sb3.sample(256,env=env)
        for a,b in zip(batch,sb3_batch):
            if a is None or b is None:
                assert a is None and b is None
            else:
                np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(batch.rewards.numpy().ravel(),raw['rewards'])
        np.testing.assert_allclose(batch.observations.numpy(),env.normalize_obs(raw['observations']))
        replay.her.rng.bit_generator.state=copy.deepcopy(rng)
        facade=IDACHERReplay(replay,'cpu');idac=facade.sample(256)
        for a,b in zip(idac,(batch.observations,batch.actions,batch.next_observations,batch.rewards,1-batch.dones)):
            np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-6,atol=1e-6)
        assert float(env.obs_rms.count)==count
        replay.her.draw=draw
        restored=pickle.loads(pickle.dumps(sb3));assert restored.her.compute_reward is None
        recovered=restored.sample(256,env=env)
        assert np.isin(recovered.rewards.numpy(),[-1.,0.]).all()
        with tempfile.TemporaryDirectory() as path:
            replay.save(Path(path)/'replay.npz');saved=np.load(Path(path)/'replay.npz')
            assert int(saved['her_completed_until'])==130 and int(saved['her_total'])==137
            assert saved['her_episode_ends'].shape==(75,)
            json.loads(str(saved['her_rng']))
        results.append(dict(task=task,future_within_episode=True,partial_episodes_excluded=True,
            ring_buffer_overwrite=True,reward_matches_environment=True,raw_storage_unchanged=True,
            relabeled_samples=204,batch_size=256,sb3_and_native_batches_identical=True,
            idac_interface_identical=True,reward_not_normalized=True,timeout_bootstrap=True,
            normalization_stats_frozen=True,pickle_and_npz_metadata=True))
        env.close()
    print(json.dumps(dict(passed=True,results=results)),flush=True)


if __name__=='__main__':main()
