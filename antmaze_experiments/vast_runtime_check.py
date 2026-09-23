"""CPU-only environment/reward checks before the queued GPU preflight."""
import json
from pathlib import Path
import numpy as np
import torch
import gym
import mujoco_py
from .envs import make_one

result=dict(torch=torch.__version__,gym=gym.__version__,numpy=np.__version__,
            mujoco_py=mujoco_py.__version__,tasks={})
assert torch.version.cuda is not None
for task in ('v1','v3','v4'):
    env=make_one(task,0,reward_profile='dense')
    obs=env.reset()
    obs,r,done,info=env.step(np.zeros(8,dtype=np.float32))
    goals=np.asarray(env.physics_env.target_goal).reshape(-1,2)
    assert np.isclose(r,-np.linalg.norm(obs[:2]-goals,axis=1).min())
    assert obs.shape==(29,) and np.isfinite(obs).all()
    env.close();result['tasks'][task]=dict(reward=float(r),passed=True)
Path('/workspace/antmaze-temperature-20260924/runtime-ready.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
