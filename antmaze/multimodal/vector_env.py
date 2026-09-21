"""256 independent native MuJoCo environments, stepped in bounded CPU threads."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .env import make_env
from .coverage import Coverage


class Collector:
    def __init__(self,task,seed,count=256,workers=8,env_factory=None):
        factory=env_factory or make_env
        self.envs=[Coverage(factory(task,seed+i)) for i in range(count)]
        self.obs=np.stack([e.reset(seed=seed+i)[0] for i,e in enumerate(self.envs)])
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.workers=workers;self.transitions=0;self.successes=0;self.episodes=0
        self.curriculum_successes=0;self.full_start_successes=0;self.falls=0
        self.full_start_episodes=0;self.curriculum_episodes=0

    def step(self,actions):
        n=len(actions);old=self.obs[:n].copy()
        def chunk(indices):
            result=[]
            for i in indices:
                e=self.envs[i];ns,r,t,tr,info=e.step(actions[i])
                reset=e.reset()[0] if t or tr else ns
                result.append((i,ns,r,t,tr,reset,info["success"],info.get("reset_kind","full_start"),info.get("fallen",False)))
            return result
        chunks=[range(i,n,self.workers) for i in range(self.workers)]
        out=[item for part in self.pool.map(chunk,chunks) for item in part]
        out.sort(key=lambda x:x[0])
        self.obs[:n]=np.stack([x[5] for x in out])
        self.transitions+=n;self.successes+=sum(x[6] for x in out)
        self.episodes+=sum(x[3] or x[4] for x in out)
        self.curriculum_successes+=sum(x[6] and x[7]=="curriculum" for x in out)
        self.full_start_successes+=sum(x[6] and x[7]=="full_start" for x in out)
        self.curriculum_episodes+=sum((x[3] or x[4]) and x[7]=="curriculum" for x in out)
        self.full_start_episodes+=sum((x[3] or x[4]) and x[7]=="full_start" for x in out)
        self.falls+=sum(x[8] for x in out)
        return old,actions,np.asarray([x[2] for x in out],np.float32),np.stack([x[1] for x in out]),np.asarray([x[3] for x in out],np.float32)

    def set_training_step(self,step):
        for e in self.envs:e.unwrapped.training_step=step

    def save(self,path):
        e=self.envs[0]
        np.savez_compressed(path,counts=sum(x.counts for x in self.envs),low=e.low,high=e.high,
            bin_size=e.bin_size,outside=sum(x.outside for x in self.envs))

    def close(self):
        self.pool.shutdown()
        for e in self.envs:e.close()
