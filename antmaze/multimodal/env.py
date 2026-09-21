"""Modern MuJoCo port of DDiffPG AntMaze geometry/robot/reset contract.

Maps and low-gear XML originate in supersglzc/ddiffpg at 7edd06c.
Reward is our explicit MaxEntDP-style nearest-goal distance objective, NOT
a claim that the unreleased MFPO AntMaze reward has been reproduced exactly.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

VENDOR=Path(__file__).resolve().parent/"vendor"
MAPS=json.loads((VENDOR/"maps.json").read_text())
GOALS={"v1":[[-8.,0.]],"v3":[[-12.,12.],[12.,-12.]],"v4":[[-16.,4.],[-16.,-4.]]}
HORIZONS={"v1":500,"v3":700,"v4":700}
SCALE=4.


def geometry(task):
    grid=MAPS[task]
    row,col=next((i,j) for i,r in enumerate(grid) for j,v in enumerate(r) if v=="r")
    walls=[[(j-col)*SCALE,(i-row)*SCALE] for i,r in enumerate(grid) for j,v in enumerate(r) if v==1]
    return dict(task=task,map=grid,scale=SCALE,walls=walls,goals=GOALS[task],start=[0.,0.],horizon=HORIZONS[task])


def model_xml(task):
    root=ET.fromstring((VENDOR/"low_gear_ant.xml").read_text())
    # MuJoCo 3 removed the legacy attribute; 'local' is already the default.
    root.find("compiler").attrib.pop("coordinate",None)
    world=root.find("worldbody")
    for site in list(world.findall("site")):world.remove(site)
    for i,(x,y) in enumerate(GOALS[task]):
        ET.SubElement(world,"site",name=f"goal{i+1}",pos=f"{x} {y} 0",size="0.5",material="target")
    for i,(x,y) in enumerate(geometry(task)["walls"]):
        ET.SubElement(world,"geom",name=f"wall{i}",pos=f"{x} {y} 1",size="2 2 1",type="box",
                      contype="1",conaffinity="1",rgba="0.7 0.5 0.3 1")
    return ET.tostring(root,encoding="unicode")


class AntMaze(gym.Env):
    metadata={"render_modes":[],"render_fps":10}
    def __init__(self, task="v1"):
        super().__init__()
        self.task=task;self.specification=geometry(task)
        xml=model_xml(task);self.xml_sha256=hashlib.sha256(xml.encode()).hexdigest()
        self.model=mujoco.MjModel.from_xml_string(xml)
        self.data=mujoco.MjData(self.model)
        self.goals=np.asarray(GOALS[task]);self.horizon=HORIZONS[task]
        self.action_space=spaces.Box(-1.,1.,shape=(8,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(29,),dtype=np.float32)
        self.initial_qpos=self.model.qpos0.copy();self.initial_qvel=np.zeros(self.model.nv)
        assert self.model.nq==15 and self.model.nv==14 and self.model.nu==8
        assert np.all(self.model.actuator_gear[:,0]==30) and self.model.opt.timestep==.02
        self.steps=0
    def observation(self):return np.concatenate([self.data.qpos[:15],self.data.qvel[:14]]).astype(np.float32)
    def simulator_state(self):
        kind=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(self.model,kind))
        mujoco.mj_getState(self.model,self.data,state,kind)
        return state
    def info(self):
        distances=np.linalg.norm(self.data.qpos[:2]-self.goals,axis=1)
        goal=int(np.argmin(distances))
        return dict(xy=self.data.qpos[:2].copy(),distance=float(distances[goal]),
            goal_id=goal+1 if distances[goal]<=.5 else 0,success=bool(distances[goal]<=.5))
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        qpos=self.initial_qpos.copy()
        # DDiffPG preprocess_cfg turns random xy reset on for v1 only.
        if self.task=="v1" and not (options or {}).get("fixed_start",False):
            qpos[:2]=self.np_random.uniform(-2,2,size=2)
        self.data.qpos[:]=qpos;self.data.qvel[:]=self.initial_qvel
        mujoco.mj_forward(self.model,self.data);self.steps=0
        return self.observation(),self.info()
    def step(self,action):
        action=np.asarray(action,np.float32)
        if action.shape!=(8,) or not np.isfinite(action).all():raise ValueError("Invalid Ant action")
        self.data.ctrl[:]=np.clip(action,-1,1)
        mujoco.mj_step(self.model,self.data,nstep=5)
        self.steps+=1;obs=self.observation()
        if not np.isfinite(obs).all():raise FloatingPointError("Nonfinite simulator state")
        info=self.info();reward=-info["distance"]
        # DDiffPG's goal wrapper discards base Ant death and locomotion rewards.
        return obs,reward,info["success"],self.steps>=self.horizon and not info["success"],info


def make_env(task="v1",seed=None):
    env=AntMaze(task)
    if seed is not None:env.action_space.seed(seed);env.observation_space.seed(seed)
    return env
