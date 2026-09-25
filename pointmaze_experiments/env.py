"""Point-mass diagnostic, not the original AntMaze benchmark."""
import gymnasium as gym
import numpy as np

class PointMaze(gym.Env):
    observation_space=gym.spaces.Box(np.array([-10,-6],np.float32),np.array([2,6],np.float32))
    action_space=gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
    def __init__(self, seed=0, random_start=True):
        self.rng=np.random.default_rng(seed);self.random_start=random_start
        self.action_space.seed(seed)
    @staticmethod
    def valid(p):
        x,y=p
        return -10<x<2 and -6<y<6 and not (-6<=x<=-2 and -2<=y<=2)
    def reset(self,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.xy=self.rng.uniform(-2,2,2) if self.random_start else np.zeros(2)
        self.steps=0
        assert self.valid(self.xy)
        return self.xy.astype(np.float32),{}
    def step(self,action):
        a=np.clip(np.asarray(action,dtype=float),-1,1)
        assert a.shape==(2,) and np.isfinite(a).all()
        # Symmetric collision handling; substeps prevent tunnelling through walls.
        for _ in range(10):
            nxt=self.xy+.02*a
            if self.valid(nxt):self.xy=nxt
        self.steps+=1
        distance=float(np.linalg.norm(self.xy-[-8,0]))
        success=distance<=.5
        return self.xy.astype(np.float32),-distance,success,self.steps>=500 and not success,dict(success=success,distance=distance)

def test_geometry():
    e=PointMaze(random_start=False);e.reset()
    for _ in range(100):e.step([-1,0])
    assert -2<e.xy[0]<-1.9 and abs(e.xy[1])<1e-8
    endpoints=[]
    for sign in [-1,1]:
        e.reset()
        for a,n in [([0,sign],20),([-1,0],40),([0,-sign],20)]:
            for _ in range(n):e.step(a)
        endpoints.append(e.xy.copy())
        assert np.linalg.norm(e.xy-[-8,0])<1e-6
    np.testing.assert_allclose(endpoints[0],endpoints[1],atol=1e-6)
