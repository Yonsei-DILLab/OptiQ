"""DDiffPG geometry with explicitly added route shaping and reset curriculum.

Only training uses this adapter. Evaluation still uses the original origin/reset
distribution and success radius in env.py, without curriculum or action guidance.
"""
from functools import lru_cache
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from .env import AntMaze, geometry


@lru_cache(maxsize=3)
def distance_field(task):
    """Eight-connected shortest distances; inflated walls and no corner cutting."""
    g=geometry(task);walls=np.asarray(g["walls"]);resolution=.25;clearance=.6
    low=walls.min(0)-2;high=walls.max(0)+2
    xs=np.arange(low[0],high[0]+resolution/2,resolution)
    ys=np.arange(low[1],high[1]+resolution/2,resolution)
    points=np.stack(np.meshgrid(xs,ys,indexing="ij"),axis=-1)
    blocked=np.any(np.all(np.abs(points[:,:,None,:]-walls)<=(2+clearance),axis=-1),axis=-1)
    free=~blocked;dist=np.full(free.shape,np.inf);queue=[]
    for goal in g["goals"]:
        ij=tuple(np.rint((np.asarray(goal)-low)/resolution).astype(int))
        assert free[ij];dist[ij]=0.;heapq.heappush(queue,(0.,*ij))
    directions=[(i,j) for i in (-1,0,1) for j in (-1,0,1) if i or j]
    while queue:
        d,i,j=heapq.heappop(queue)
        if d>dist[i,j]:continue
        for di,dj in directions:
            x,y=i+di,j+dj
            if not(0<=x<free.shape[0] and 0<=y<free.shape[1]) or not free[x,y]:continue
            if di and dj and (not free[i+di,j] or not free[i,j+dj]):continue
            nd=d+resolution*np.hypot(di,dj)
            if nd<dist[x,y]:dist[x,y]=nd;heapq.heappush(queue,(nd,x,y))
    reachable=np.isfinite(dist)
    assert reachable.any()
    # Extend the field inside the safety margin by distance to the nearest
    # reachable cell. This keeps reward finite for the native random reset.
    offset,nearest=distance_transform_edt(~reachable,return_indices=True)
    extended=dist[tuple(nearest)]+offset*resolution
    if task=="v1":
        # EDT breaks equidistant-cell ties by index inside walls. Remove that
        # arbitrary upper/lower preference on this reflection-symmetric maze.
        assert np.array_equal(reachable,reachable[:,::-1])
        extended=.5*(extended+extended[:,::-1])
    starts=points[reachable & (dist>=1.)]
    start_distances=dist[reachable & (dist>=1.)]
    return dict(low=low,resolution=resolution,values=extended,free=free,
                starts=starts,start_distances=start_distances,clearance=clearance)


def route_distance(task,xy):
    field=distance_field(task)
    ij=(np.asarray(xy)-field["low"])/field["resolution"]
    return float(map_coordinates(field["values"],ij[:,None],order=1,mode="nearest")[0])


class RouteTrainingEnv(AntMaze):
    """Shared training assistance; both routes receive symmetric support."""
    def __init__(self,task="v1"):
        super().__init__(task)
        self.training_step=0;self.reset_kind="full_start"
        self.field=distance_field(task)
        self.origin_distance=route_distance(task,np.zeros(2))
    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        # Expand backward from the goal through 60k, taper assistance by 80k.
        p_full=.2 if self.training_step<60000 else (.75 if self.training_step<80000 else 1.)
        self.reset_kind="full_start"
        if self.np_random.random()>=p_full:
            limit=2.+(self.origin_distance-2.)*min(self.training_step/60000,1.)
            mask=self.field["start_distances"]<=limit
            choices=self.field["starts"][mask]
            assert len(choices)
            self.data.qpos[:2]=choices[self.np_random.integers(len(choices))]
            import mujoco
            mujoco.mj_forward(self.model,self.data)
            self.reset_kind="curriculum"
            obs=self.observation();info=self.info()
        info["reset_kind"]=self.reset_kind
        info["route_distance"]=route_distance(self.task,info["xy"])
        return obs,info
    def step(self,action):
        previous=route_distance(self.task,self.data.qpos[:2])
        obs,_,success,truncated,info=super().step(action)
        current=route_distance(self.task,info["xy"])
        # Progress measured in m/s (Ant frame skip 5, dt .1). Added locomotion
        # terms favor useful upright motion rather than dragging along walls.
        progress=(previous-current)/.1
        healthy=.2<=self.data.qpos[2]<=1.
        control=.05*float(np.square(np.clip(action,-1,1)).sum())
        reward=progress+(.1 if healthy else 0.)-control-.01+10.*success
        fallen=not healthy and not success
        if fallen:reward-=5.
        info.update(route_distance=current,route_progress=progress,control_cost=control,
                    healthy=healthy,fallen=fallen,reset_kind=self.reset_kind,
                    training_reward=reward)
        return obs,reward,success or fallen,truncated and not fallen,info


def make_route_env(task="v1",seed=None):
    env=RouteTrainingEnv(task)
    if seed is not None:env.action_space.seed(seed)
    return env
