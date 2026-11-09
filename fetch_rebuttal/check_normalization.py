"""Executable consistency checks for the common Fetch normalization adapters."""
import json
from pathlib import Path
import tempfile
import numpy as np
from .normalization import make_normalized_env, save_normalizer, NormalizedIDACReplay
from .baselines.replay import Replay


def main():
    rows=[]
    for task, dim in [('pick-and-place',28),('push',28),('slide',28),('reach',13)]:
        env=make_normalized_env(task);env.seed(123);obs=env.reset()
        replay=Replay(capacity=100,observation_dim=dim,action_dim=4,device='cpu')
        replay.normalizer=env;raw=env.get_original_obs()
        for step in range(50):
            obs,reward,done,infos=env.step(np.zeros((1,4),np.float32))
            nxt=np.asarray([infos[0]['fetch_raw_next_observation']])
            terminal=np.array([done[0] and not infos[0].get('TimeLimit.truncated',False)],np.float32)
            replay.add(raw,np.zeros((1,4),np.float32),reward,nxt,terminal)
            raw=env.get_original_obs()
        assert done[0] and not terminal[0] and infos[0]['TimeLimit.truncated']
        assert np.array_equal(replay.data['next_observations'][49],nxt[0])
        assert np.isin(replay.data['rewards'][:50],[-1.,0.]).all()
        assert not replay.data['terminals'][:50].any()
        old_count=float(env.obs_rms.count)
        probe=replay.data['observations'][:4].copy()
        expected=env.normalize_obs(probe)
        assert float(env.obs_rms.count)==old_count
        # Fix the sampled indices, ensuring both sampled states use current RMS.
        state=replay.rng.bit_generator.state
        indices=replay.rng.integers(replay.size,size=32)
        replay.rng.bit_generator.state=state
        batch=replay.sample(32)
        np.testing.assert_allclose(batch.observations.numpy(),env.normalize_obs(replay.data['observations'][indices]))
        np.testing.assert_allclose(batch.next_observations.numpy(),env.normalize_obs(replay.data['next_observations'][indices]))
        np.testing.assert_equal(batch.rewards.numpy().ravel(),replay.data['rewards'][indices])
        class Buffer:pass
        r=Buffer();r.size=50;r.state=replay.data['observations'][:50];r.next_state=replay.data['next_observations'][:50]
        r.action=replay.data['actions'][:50];r.reward=replay.data['rewards'][:50,None]
        r.not_done=1-replay.data['terminals'][:50,None];r.device='cpu'
        proxy=NormalizedIDACReplay(r,env)
        np.random.seed(5);idx=np.random.randint(0,50,size=32);np.random.seed(5)
        data=proxy.sample(32)
        np.testing.assert_allclose(data[0].numpy(),env.normalize_obs(r.state[idx]))
        np.testing.assert_allclose(data[2].numpy(),env.normalize_obs(r.next_state[idx]))
        assert float(env.obs_rms.count)==old_count
        with tempfile.TemporaryDirectory() as d:
            save_normalizer(env,d,50)
            from stable_baselines3.common.vec_env import VecNormalize
            loaded=VecNormalize.load(str(Path(d)/'normalizer_000000050.pkl'),env.venv)
            loaded.training=False
            np.testing.assert_array_equal(loaded.normalize_obs(probe),expected)
        rows.append(dict(task=task,shape=dim,raw_replay=True,timeout_bootstrap=True,
            identical_sample_normalization=True,evaluation_stats_frozen=True,checkpoint_roundtrip=True))
        env.close()
    print(json.dumps({'passed':True,'checks':rows}),flush=True)


if __name__=='__main__':main()
