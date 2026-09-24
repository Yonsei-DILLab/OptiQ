"""Distance progress + time cost + upstream success bonus (no simulator dependency).

Geodesic = continuous XY point-agent shortest path around axis-aligned maze
walls, via a visibility graph. This is not a MuJoCo configuration-space path
or Habitat NavMesh. A 1e-6 m wall margin prevents shared-edge/corner shortcuts.
"""
import ast
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np

LEGACY_PROFILES = ('progress_euclidean', 'progress_geodesic',
            'progress_euclidean_no_bonus', 'progress_geodesic_no_bonus')

SCALED_PROFILES = tuple(p.replace('progress_', 'progress100_') for p in LEGACY_PROFILES)
PROFILES = LEGACY_PROFILES + SCALED_PROFILES

def progress_scale(profile):
    if profile not in PROFILES: raise ValueError(profile)
    return 100. if profile in SCALED_PROFILES else 1.

def step_cost(profile):
    return STEP_COST * progress_scale(profile)

def bonus_enabled(profile):
    if profile not in PROFILES: raise ValueError(profile)
    return not profile.endswith('_no_bonus')

def is_geodesic(profile):
    return profile in PROFILES and 'geodesic' in profile
STEP_COST = .01
GEOMETRY_MARGIN = 1e-6
UPSTREAM = Path(__file__).resolve().parents[1]/'antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py'


@lru_cache(None)
def maze_geometry(task):
    if task not in ('v1','v2','v3','v4'):
        raise ValueError(task)
    class Markers(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Constant({'R':'r','G':'g'}[node.id])
    tree = ast.parse(UPSTREAM.read_text())
    node = next(n for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='MAZE_'+task)
    maze = ast.literal_eval(Markers().visit(node.value))
    ri,ci = next((i,j) for i,row in enumerate(maze) for j,v in enumerate(row) if v=='r')
    walls=[];goals=[]
    for i,row in enumerate(maze):
        for j,v in enumerate(row):
            x,y=(j-ci)*4.,(i-ri)*4.
            if v==1:walls.append([x-2,y-2,x+2,y+2])
            if v=='g':goals.append([x,y])
    goals.sort(key=lambda p:(p[0],-p[1]))  # Match upstream goal_sampler IDs in v1-v4.
    bounds=np.array([-ci*4-2,-ri*4-2,(len(maze[0])-1-ci)*4+2,(len(maze)-1-ri)*4+2])
    return np.asarray(walls),np.asarray(goals),bounds


class Geodesic:
    """Shortest path to each goal; immutable graph shared by forked env workers."""
    def __init__(self, walls, goals, bounds=None, margin=GEOMETRY_MARGIN):
        self.walls=np.asarray(walls,dtype=np.float64).reshape(-1,4).copy()
        self.walls[:,:2]-=margin;self.walls[:,2:]+=margin
        self.goals=np.asarray(goals,dtype=np.float64).reshape(-1,2).copy()
        self.bounds=None if bounds is None else np.asarray(bounds,dtype=np.float64)
        corners=self.walls[:,[0,1,2,1,2,3,0,3]].reshape(-1,2)
        vertices=np.unique(np.concatenate((self.goals,corners)),axis=0)
        vertices=vertices[self.valid(vertices)]
        self.vertices=vertices
        for goal in self.goals:
            if not np.any(np.all(vertices==goal,axis=1)):raise ValueError('Goal inside obstacle')
        matrix=np.linalg.norm(vertices[:,None,:]-vertices[None,:,:],axis=-1)
        for i,p in enumerate(vertices):matrix[i,~self.visible(p,vertices)]=np.inf
        np.fill_diagonal(matrix,0.)
        for k in range(len(vertices)):
            matrix=np.minimum(matrix,matrix[:,k,None]+matrix[None,k,:])
        indices=[np.flatnonzero(np.all(vertices==g,axis=1))[0] for g in self.goals]
        self.to_goals=matrix[:,indices]
        # Reachable shortest paths have a visible vertex/goal; distance to it is
        # <= domain diagonal. This conservative bound also bounds progress returns.
        diagonal=np.linalg.norm(self.bounds[2:]-self.bounds[:2]) if bounds is not None else 0.
        finite=self.to_goals[np.isfinite(self.to_goals)]
        self.distance_bound=float(diagonal+(finite.max() if finite.size else 0.))
        for value in (self.walls,self.goals,self.vertices,self.to_goals):value.flags.writeable=False

    def valid(self, points):
        p=np.asarray(points,dtype=np.float64).reshape(-1,2)
        inside=((p[:,None,:]>self.walls[None,:,:2]+1e-10)&
                (p[:,None,:]<self.walls[None,:,2:]-1e-10)).all(-1).any(-1)
        valid=np.isfinite(p).all(-1)&~inside
        if self.bounds is not None:valid&=((p>self.bounds[:2])&(p<self.bounds[2:])).all(-1)
        return valid

    def visible(self, point, ends):
        """Reject intersection with any rectangle interior; boundary tangency OK."""
        point=np.asarray(point);delta=np.asarray(ends)-point
        if not len(self.walls):return np.ones(len(delta),bool)
        lo=self.walls[None,:,:2]-point;hi=self.walls[None,:,2:]-point
        direction=delta[:,None,:];parallel=np.abs(direction)<1e-14
        divisor=np.where(parallel,1.,direction)
        t1,t2=lo/divisor,hi/divisor
        enter,leave=np.minimum(t1,t2),np.maximum(t1,t2)
        strictly_inside=(point>self.walls[:,:2]+1e-10)&(point<self.walls[:,2:]-1e-10)
        enter=np.where(parallel,np.where(strictly_inside[None,:,:],-np.inf,np.inf),enter)
        leave=np.where(parallel,np.where(strictly_inside[None,:,:],np.inf,-np.inf),leave)
        entry=np.maximum(enter.max(-1),0.)
        exit_=np.minimum(leave.min(-1),1.)
        return ~((exit_-entry)>1e-10).any(-1)

    def distances(self, points):
        p=np.asarray(points,dtype=np.float64)
        shape=p.shape[:-1];flat=p.reshape(-1,2)
        valid=self.valid(flat)
        if not valid.all():raise ValueError(f'Geodesic query outside free XY space: {flat[~valid][:3]}')
        out=np.empty((len(flat),len(self.goals)))
        for i,point in enumerate(flat):
            costs=np.linalg.norm(self.vertices-point,axis=-1)
            costs[~self.visible(point,self.vertices)]=np.inf
            out[i]=(costs[:,None]+self.to_goals).min(axis=0)
        if not np.isfinite(out).all():raise ValueError('Goal is disconnected from query position')
        return out.reshape(shape+(len(self.goals),))


@lru_cache(None)
def geodesic(task):
    return Geodesic(*maze_geometry(task))


def distance(points, task, profile):
    if profile not in PROFILES:raise ValueError(profile)
    points=np.asarray(points,dtype=np.float64)
    if is_geodesic(profile):return geodesic(task).distances(points).min(axis=-1)
    goals=maze_geometry(task)[1]
    return np.linalg.norm(points[...,None,:]-goals,axis=-1).min(axis=-1)


def success_bonus(points, task):
    points=np.asarray(points,dtype=np.float64);goals=maze_geometry(task)[1]
    distances=np.linalg.norm(points[...,None,:]-goals,axis=-1)
    reached=distances<=.5
    values=np.array([20. if tuple(g)==(-8.,8.) else 10. for g in goals])
    # First upstream goal in goal_sampler order wins, as in check_goal().
    return np.where(reached.any(axis=-1),values[reached.argmax(axis=-1)],0.)


def progress_reward(before, after, task, profile, bonus):
    previous=distance(before,task,profile);current=distance(after,task,profile)
    return progress_scale(profile)*(previous-current)-step_cost(profile)+(np.asarray(bonus) if bonus_enabled(profile) else 0.),previous,current


def specification(task, profile):
    if profile not in PROFILES:return None
    walls,goals,bounds=maze_geometry(task)
    formula=('100*(d(current)-d(next))-1' if profile in SCALED_PROFILES else 'd(current)-d(next)-0.01')
    return dict(formula=formula+('+upstream_success_bonus' if bonus_enabled(profile) else ''),
        success_bonus_enabled=bonus_enabled(profile),
        distance='nearest-goal Euclidean' if not is_geodesic(profile) else 'nearest-goal XY visibility-graph geodesic',
        step_cost=step_cost(profile),progress_scale=progress_scale(profile),discount_inside_reward=False,goals=goals.tolist(),
        goal_bonuses=[(20 if tuple(g)==(-8,8) else 10) if bonus_enabled(profile) else 0 for g in goals],
        distance_endpoint='goal center; do not replace terminal distance by zero',
        success_radius=.5,success_terminates=True,timeout_bootstraps=True,
        geometry_margin_m=GEOMETRY_MARGIN if is_geodesic(profile) else 0.,
        physical_body_inflation_m=0.,map_scale_m=4.,
        upstream_geometry_sha256=hashlib.sha256(UPSTREAM.read_bytes()).hexdigest())


def value_support(task, profile, gamma=.99):
    """Conservative C51 bounds for this reward, not the old sparse [0,5].

Discounted progress is bounded by +/-progress_scale*Dmax. A terminating
goal adds at most20 once; time cost is -step_cost/(1-gamma).
    """
    if profile not in PROFILES:raise ValueError(profile)
    walls,goals,bounds=maze_geometry(task)
    dmax=geodesic(task).distance_bound if is_geodesic(profile) else float(np.linalg.norm(bounds[2:]-bounds[:2]))
    return -float(np.ceil(progress_scale(profile)*dmax+step_cost(profile)/(1-gamma)+1)),float(np.ceil(progress_scale(profile)*dmax+20+1))
