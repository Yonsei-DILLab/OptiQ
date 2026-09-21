"""Training xy occupancy only; evaluation visits never enter these counts."""
import gymnasium as gym
import numpy as np
from antmaze.evaluation import atomic_json


class Coverage(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        walls=np.asarray(env.specification["walls"])
        self.low=walls.min(0)-2;self.high=walls.max(0)+2;self.bin_size=.5
        shape=np.ceil((self.high-self.low)/self.bin_size).astype(int)
        self.counts=np.zeros(tuple(shape),np.int64);self.outside=0
        self.total_steps=0;self.episodes=[];self.successes=0;self.first_success_step=None
        self.min_distance=float("inf");self.episode_length=0;self.episode_return=0.
        self.episode_min_distance=float("inf")
    def step(self,action):
        result=self.env.step(action);xy=result[-1]["xy"]
        ij=np.floor((xy-self.low)/self.bin_size).astype(int)
        if (ij>=0).all() and (ij<self.counts.shape).all():self.counts[tuple(ij)]+=1
        else:self.outside+=1
        _,reward,terminated,truncated,info=result
        self.total_steps+=1;self.episode_length+=1;self.episode_return+=float(reward)
        distance=float(info["distance"])
        self.min_distance=min(self.min_distance,distance)
        self.episode_min_distance=min(self.episode_min_distance,distance)
        if terminated or truncated:
            success=bool(info["success"])
            self.successes+=int(success)
            if success and self.first_success_step is None:self.first_success_step=self.total_steps
            self.episodes.append(dict(end_step=self.total_steps,length=self.episode_length,
                episode_return=self.episode_return,min_distance=self.episode_min_distance,
                final_distance=distance,success=success,goal_id=int(info["goal_id"]),
                terminated=bool(terminated),truncated=bool(truncated)))
            self.episode_length=0;self.episode_return=0.;self.episode_min_distance=float("inf")
        return result
    def training_summary(self):
        return dict(training_steps=self.total_steps,training_episodes=len(self.episodes),
            training_successes=self.successes,training_min_distance=self.min_distance if self.total_steps else None,
            first_success_step=self.first_success_step)
    def save(self,path):
        np.savez_compressed(path,counts=self.counts,low=self.low,high=self.high,
                            bin_size=self.bin_size,outside=self.outside)
        atomic_json(path.with_name("training_episodes.json"),dict(**self.training_summary(),episodes=self.episodes))
