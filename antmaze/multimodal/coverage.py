"""Training xy occupancy only; evaluation visits never enter these counts."""
import gymnasium as gym
import numpy as np


class Coverage(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        walls=np.asarray(env.specification["walls"])
        self.low=walls.min(0)-2;self.high=walls.max(0)+2;self.bin_size=.5
        shape=np.ceil((self.high-self.low)/self.bin_size).astype(int)
        self.counts=np.zeros(tuple(shape),np.int64);self.outside=0
    def step(self,action):
        result=self.env.step(action);xy=result[-1]["xy"]
        ij=np.floor((xy-self.low)/self.bin_size).astype(int)
        if (ij>=0).all() and (ij<self.counts.shape).all():self.counts[tuple(ij)]+=1
        else:self.outside+=1
        return result
    def save(self,path):
        np.savez_compressed(path,counts=self.counts,low=self.low,high=self.high,
                            bin_size=self.bin_size,outside=self.outside)
