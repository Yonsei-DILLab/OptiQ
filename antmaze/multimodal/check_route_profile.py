"""Deterministic checks of the training assistance and untouched evaluation."""
import json
import numpy as np
from .env import make_env
from .route_env import make_route_env,route_distance,distance_field


def main():
    d=route_distance("v1",[0.,0.])
    assert d>11.,"Distance must go around the central obstacle"
    for x in np.linspace(-8,0,17):
        for y in np.linspace(0,4,9):
            np.testing.assert_allclose(route_distance("v1",[x,y]),route_distance("v1",[x,-y]),atol=1e-10)
    field=distance_field("v1")
    for sign in (-1,1):
        assert route_distance("v1",[0,sign*1.])<d
        assert route_distance("v1",[-4,sign*4.])<route_distance("v1",[0,sign*4.])
    env=make_route_env();evaluation=make_env();kinds=[];points=[]
    for seed in range(100):
        obs,info=env.reset(seed=seed);kinds.append(info["reset_kind"])
        if info["reset_kind"]=="curriculum":
            assert 1.-1e-8<=info["route_distance"]<=2.+1e-8
            xy=info["xy"];points.append(xy)
            assert np.all(np.any(np.abs(xy-np.asarray(env.specification["walls"]))>2.6-1e-8,axis=1))
    assert kinds.count("full_start")>=10 and kinds.count("curriculum")>=60
    assert min(p[1] for p in points)<0<max(p[1] for p in points)
    env.training_step=80000
    for seed in range(30):
        obs,info=env.reset(seed=seed);expected,_=evaluation.reset(seed=seed)
        assert info["reset_kind"]=="full_start"
        np.testing.assert_array_equal(obs,expected)
    env.reset(seed=1);env.data.qpos[:2]=[-8,0]
    _,r,terminal,truncated,info=env.step(np.zeros(8))
    assert terminal and not truncated and info["success"] and r>9.
    env.reset(seed=2);env.data.qpos[2]=.1
    _,r,terminal,_,info=env.step(np.zeros(8))
    assert terminal and info["fallen"] and not info["success"]
    env.close();evaluation.close()
    print(json.dumps(dict(passed=True,origin_route_distance=d,reset_counts={k:kinds.count(k) for k in set(kinds)},
        tests=["wall_detour","upper_lower_symmetry","curriculum_clearance","full_start_after_80k", "success_and_fall_terminals"])))


if __name__=="__main__":main()
