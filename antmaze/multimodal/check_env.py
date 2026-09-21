"""Validate port geometry, dense reward, goal termination and equal simulator state."""
import numpy as np
import mujoco
from .env import make_env,GOALS,HORIZONS,geometry
from .analysis import route_label


def main():
    for task in ("v1","v3","v4"):
        a,b=make_env(task,0),make_env(task,0)
        x,_=a.reset(seed=11,options={"fixed_start":True})
        y,_=b.reset(seed=999,options={"fixed_start":True})
        assert x.shape==(29,) and a.action_space.shape==(8,)
        np.testing.assert_array_equal(a.simulator_state(),b.simulator_state())
        for i in range(HORIZONS[task]):
            x,r,t,tr,info=a.step(np.zeros(8))
            assert r==-info["distance"] and np.isfinite(x).all()
            if t or tr:break
        assert tr and not t and i+1==HORIZONS[task]
        for gid,g in enumerate(GOALS[task],1):
            a.reset(options={"fixed_start":True})
            a.data.qpos[:2]=g;mujoco.mj_forward(a.model,a.data)
            _,r,t,tr,info=a.step(np.zeros(8))
            assert t and not tr and info["goal_id"]==gid and -.5<=r<=0
        # Every declared goal is a free map cell and start is at origin.
        geo=geometry(task)
        for g in GOALS[task]+[[0,0]]:
            assert not any(np.all(np.abs(np.asarray(g)-w)<2) for w in geo["walls"])
        a.close();b.close()
    assert route_label("v1",np.array([[0,0],[0,4],[-8,4],[-8,0]]),1)=="G1/upper"
    assert route_label("v1",np.array([[0,0],[0,-4],[-8,-4],[-8,0]]),1)=="G1/lower"
    assert route_label("v4",np.array([[0,0],[0,4],[-8,4],[-8,8],[-16,8],[-16,4]]),1)=="G1/upper-entry/upper-outer"
    assert route_label("v4",np.array([[0,0],[0,-4],[-8,-4],[-8,0],[-16,0],[-16,-4]]),2)=="G2/lower-entry/middle"
    print("Passed v1/v3/v4 physics,29D state,goals,dense reward,timeouts,fixed-full-state,route labels")


if __name__=="__main__":main()
