"""One real GPU update and native inference for every Adroit baseline shape."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from .run import TASKS, atomic, guard, make_env
from .baseline_run import finite


def main():
    guard()
    import torch
    torch.set_num_threads(1)
    from .baselines.adapters import create, action, DIPOReplay
    from .demo_replay import DemonstrationReplay
    root = Path(os.environ.get('ADROIT_VALIDATION_ROOT', '/workspace/heechan-adroit/followup'));root.mkdir(exist_ok=True)
    results = []
    registry = json.loads(Path(os.environ['ADROIT_DEMO_REGISTRY']).read_text()) if os.environ.get('ADROIT_DEMO_REGISTRY') else {}
    for method in ('dipo', 'meow', 'mfpo'):
        for task in TASKS:
            np.random.seed(831);random.seed(831);torch.manual_seed(831)
            env = make_env(task)
            try:
                from .verified_demos import resolve
                demo_path, proof = resolve(task, registry.get(task, {}))
                rc = DIPOReplay if method == 'dipo' else DemonstrationReplay
                replay = rc(32768, env.observation_space, env.action_space, n_envs=256, device='cpu',
                    demo_path=str(demo_path), demo_seed=831)
                obs, nxt, actions, rewards, dones, infos = [], [], [], [], [], []
                env.action_space.seed(831)
                for i in range(256):
                    x, _ = env.reset(seed=831+i);a=env.action_space.sample()
                    y,r,t,tr,info=env.step(a)
                    obs.append(x);nxt.append(y);actions.append(a);rewards.append(r);dones.append(t or tr)
                    infos.append(dict(info, **{'TimeLimit.truncated': tr and not t}))
                replay.add(np.asarray(obs),np.asarray(nxt),np.asarray(actions),np.asarray(rewards),np.asarray(dones),infos)
                agent = create(method,831,root,5000192,env.observation_space.shape[0],env.action_space,replay)
                from .baselines.architecture import audit
                architecture = audit(agent,method)
                original = replay.demo['actions'].copy()
                if method=='dipo':
                    memory=agent.native.diffusion_memory
                    _,labels,ids=memory.sample(256)
                    replacement=np.full((256,env.action_space.shape[0]),.125,np.float32)
                    memory.replace(ids,replacement)
                    for idx in ids:
                        assert np.all((replay.best_offline[-1-idx] if idx<0 else replay.best_online[idx//256,idx%256])==.125)
                    replay.best_offline[:]=replay.demo['actions']
                    replay.best_online[0]=np.asarray(actions)
                agent.update(13056)
                assert audit(agent,method) == architecture
                assert agent.updates==1 and finite(agent,method)
                assert np.array_equal(original,replay.demo['actions'])
                assert replay.last_mix['per_optimizer_minibatch']==[128,128]
                with agent.evaluation_rng(4123):a=action(agent,method,np.asarray(obs[:1],np.float32),evaluation=True)
                with agent.evaluation_rng(4123):b=action(agent,method,np.asarray(obs[:1],np.float32),evaluation=True)
                assert a.shape==(1,env.action_space.shape[0]) and np.isfinite(a).all()
                assert np.allclose(a,b,rtol=1e-5,atol=1e-6)
                result=dict(method=method,task=task,passed=True,updates=1,action_dim=a.shape[1],
                    native_inference_repeatable=True,immutable_original_demo_actions=True,
                    architecture=architecture,
                    demonstration_dataset_sha256=proof['dataset_sha256'] if proof else None,
                    exact_replay_mix=[128,128],persistent_dipo_labels=method=='dipo')
                results.append(result)
                atomic(root/'compatibility.json',dict(passed=len(results)==12,checks=results))
                print(json.dumps(result),flush=True)
                del agent,replay
            finally:
                env.close()


if __name__=='__main__':main()
