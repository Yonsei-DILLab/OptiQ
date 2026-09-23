"""Check evaluation-only random starts without changing training defaults."""
import json
import numpy as np
from .envs import make_one


def main():
    results={}
    for task in ('v1','v2','v3','v4'):
        train=make_one(task,123,reward_profile='dense')
        evaluation=make_one(task,123,reward_profile='dense',random_init=True)
        duplicate=make_one(task,123,reward_profile='dense',random_init=True)
        try:
            train_states=np.array([train.reset() for _ in range(100)])
            states=np.array([evaluation.reset() for _ in range(100)])
            repeated=np.array([duplicate.reset() for _ in range(100)])
            assert np.array_equal(states,repeated)
            assert np.all((states[:,:2]>=-2)&(states[:,:2]<=2))
            assert len(np.unique(states[:,:2],axis=0))==100
            assert np.array_equal(states[:,2:],np.repeat(states[:1,2:],100,axis=0))
            train_unique=len(np.unique(train_states[:,:2],axis=0))
            assert train_unique==(100 if task=='v1' else 1)
            results[task]=dict(training_unique_starts=train_unique,evaluation_unique_starts=100,
                same_pose_velocity=True,reproducible_from_seed=True,
                xy_min=states[:,:2].min(0).tolist(),xy_max=states[:,:2].max(0).tolist())
        finally:train.close();evaluation.close();duplicate.close()
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
