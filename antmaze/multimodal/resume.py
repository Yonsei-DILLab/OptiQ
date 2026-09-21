"""Complete, atomic continuation snapshots for the dense+NovelD campaign.

Only load trusted snapshots produced by this experiment: state.pt contains
optimizer and RNG objects, not merely inference weights.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from antmaze.evaluation import atomic_json


def digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        elif isinstance(x,torch.Tensor):visit(x.detach().cpu().numpy())
        elif hasattr(x,"shape") and hasattr(x,"dtype"):
            a=np.asarray(x);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        elif isinstance(x,bytes):h.update(x)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def model_state(agent):
    from antmaze.agents import SB3,MEOW
    import flax.serialization as fs
    if isinstance(agent,SB3):
        m=agent.model
        common=dict(updates=m._n_updates,num_timesteps=m.num_timesteps,
                    progress=m._current_progress_remaining)
        if agent.method=="sac":
            from stable_baselines3.common.save_util import recursive_getattr
            _,names=m._get_torch_save_params()
            return dict(**common,parameters=m.get_parameters(),
                        variables={n:recursive_getattr(m,n) for n in names})
        p=m.policy
        return dict(**common,policy={k:fs.to_bytes(getattr(p,k)) for k in
                ("actor_state","target_actor_state","qf_state","key","noise_key")},
            model={k:fs.to_bytes(getattr(m,k)) for k in
                ("ent_coef_state","key","regulator_log_alpha","regulator_state","regulator_key")},
            regulator={k:copy.deepcopy(getattr(m,k)) for k in
                ("regulator_next_update","regulator_count","regulator_entropy")},
            regulator_rng=copy.deepcopy(m.regulator_rng.bit_generator.state))
    if isinstance(agent,MEOW):
        return dict(updates=agent.updates,policy=agent.policy.state_dict(),
                    target=agent.target.state_dict(),optimizer=agent.optimizer.state_dict())
    return dict(updates=agent.updates,agent=fs.to_bytes(agent.agent))


def restore_model(agent,state):
    from antmaze.agents import SB3,MEOW
    import flax.serialization as fs
    if isinstance(agent,SB3):
        m=agent.model
        m._n_updates=state["updates"];m.num_timesteps=state["num_timesteps"]
        m._current_progress_remaining=state["progress"]
        if agent.method=="sac":
            from stable_baselines3.common.save_util import recursive_getattr
            import torch
            m.set_parameters(state["parameters"],exact_match=True,device="cuda")
            with torch.no_grad():
                for k,v in state["variables"].items():recursive_getattr(m,k).copy_(v)
        else:
            for obj,group in ((m.policy,"policy"),(m,"model")):
                for k,v in state[group].items():setattr(obj,k,fs.from_bytes(getattr(obj,k),v))
            for k,v in state["regulator"].items():setattr(m,k,v)
            m.regulator_rng.bit_generator.state=state["regulator_rng"]
    elif isinstance(agent,MEOW):
        for k in ("policy","target","optimizer"):getattr(agent,k).load_state_dict(state[k])
        agent.updates=state["updates"]
    else:
        agent.agent=fs.from_bytes(agent.agent,state["agent"]);agent.updates=state["updates"]


def snapshot(agent,replay,coverage,behavior_rng,evaluator,extra):
    import torch
    e=coverage.unwrapped;n=replay.intrinsic
    return dict(model=model_state(agent),
        intrinsic=dict(predictor=n.predictor.state_dict(),target=n.target.state_dict(),
            optimizer=n.optimizer.state_dict(),updates=n.updates,metrics=n.metrics,config=n.config),
        replay=dict(capacity=replay.capacity,size=replay.size,position=replay.position,
            rng=copy.deepcopy(replay.rng.bit_generator.state),diagnostic=replay.diagnostic),
        environment=dict(simulator=e.simulator_state(),steps=e.steps,
            rng=copy.deepcopy(e.np_random.bit_generator.state),xml_sha256=e.xml_sha256),
        coverage={k:copy.deepcopy(getattr(coverage,k)) for k in ("counts","outside","total_steps",
            "episodes","successes","first_success_step","min_distance","episode_length",
            "episode_return","episode_min_distance")},
        behavior_rng=copy.deepcopy(behavior_rng.bit_generator.state),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()),histories=copy.deepcopy(evaluator.histories),extra=extra)


def save(folder,agent,replay,coverage,behavior_rng,evaluator,extra):
    import torch
    folder=Path(folder);folder.parent.mkdir(parents=True,exist_ok=True)
    temp=folder.with_name(folder.name+".tmp")
    temp.mkdir(exist_ok=False)
    state=snapshot(agent,replay,coverage,behavior_rng,evaluator,extra)
    torch.save(state,temp/"state.pt")
    np.savez(temp/"replay.npz",**{k:v[:replay.size] for k,v in replay.data.items()})
    proof=dict(step=coverage.total_steps,updates=agent.updates,source_commit=extra["source_commit"],
        state_digest=digest(state),replay_digest=digest({k:v[:replay.size] for k,v in replay.data.items()}),
        replay_size=replay.size,files={})
    for name in ("state.pt","replay.npz"):
        p=temp/name
        with p.open("rb") as f:
            h=hashlib.file_digest(f,"sha256").hexdigest()
        proof["files"][name]=dict(bytes=p.stat().st_size,sha256=h)
    atomic_json(temp/"manifest.json",proof)
    os.rename(temp,folder)
    atomic_json(folder.parent/"latest.json",dict(path=str(folder.resolve()),**proof))
    return proof


def load(folder,agent,replay,coverage,behavior_rng,evaluator,source_commit,task,method):
    import torch
    import mujoco
    folder=Path(folder);proof=json.loads((folder/"manifest.json").read_text())
    assert proof["source_commit"]==source_commit
    for name,spec in proof["files"].items():
        with (folder/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==spec["sha256"]
    state=torch.load(folder/"state.pt",map_location="cpu",weights_only=False)
    assert digest(state)==proof["state_digest"]
    assert state["extra"]["task"]==task and state["extra"]["method"]==method
    restore_model(agent,state["model"])
    n=replay.intrinsic
    assert n.config==state["intrinsic"]["config"]
    for k in ("predictor","target","optimizer"):getattr(n,k).load_state_dict(state["intrinsic"][k])
    n.updates=state["intrinsic"]["updates"];n.metrics=state["intrinsic"]["metrics"]
    meta=state["replay"];assert replay.capacity==meta["capacity"]
    replay.size=meta["size"];replay.position=meta["position"];replay.diagnostic=meta["diagnostic"]
    replay.rng.bit_generator.state=meta["rng"]
    with np.load(folder/"replay.npz") as data:
        for k,v in replay.data.items():v[:replay.size]=data[k]
    assert digest({k:v[:replay.size] for k,v in replay.data.items()})==proof["replay_digest"]
    e=coverage.unwrapped;es=state["environment"];assert e.xml_sha256==es["xml_sha256"]
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(e.model,e.data,es["simulator"],kind);mujoco.mj_forward(e.model,e.data)
    mujoco.mj_setState(e.model,e.data,es["simulator"],kind)
    e.steps=es["steps"];e.np_random.bit_generator.state=es["rng"]
    for k,v in state["coverage"].items():setattr(coverage,k,v)
    behavior_rng.bit_generator.state=state["behavior_rng"];evaluator.histories=state["histories"]
    random.setstate(state["rng"]["python"]);np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"]);torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    restored=snapshot(agent,replay,coverage,behavior_rng,evaluator,state["extra"])
    assert digest(restored)==proof["state_digest"],"Loaded state differs from saved training state"
    return state["extra"],proof
