"""One online AntMaze run. --smoke is a separate, never reported preflight run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
from .env import make_env, ENV_ID, ENV_KWARGS, OBS_KEYS
from .evaluation import Evaluator, atomic_json


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--method", required=True, choices=["optiq","sac","meow","sql","mfpo","dipo"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0)); gpu = int(os.environ.get("CAMPAIGN_GPU", "0"))
    os.sched_setaffinity(0, cpus[gpu::4] or cpus)
    import torch
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed); np.random.seed(args.seed)
    folder = args.output; folder.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip()
    if dirty: raise RuntimeError("Commit tracked source changes before launching")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if args.method in ("optiq","mfpo","sql"):
        import jax
        if jax.default_backend() != "gpu": raise RuntimeError("JAX GPU required")
    from .agents import DIPO, MEOW, MFPO, SQL, SB3, parameter_audit
    env = make_env(args.seed)
    if args.method in ("optiq","sac"):
        agent = SB3(args.method, env, args.seed, folder, args.smoke)
    else:
        agent = dict(dipo=DIPO, meow=MEOW, mfpo=MFPO, sql=SQL)[args.method](env,args.seed,folder)
    warmup = 256 if args.smoke else agent.warmup
    initial_parameters = parameter_audit(agent)
    steps = 272 if args.smoke else 1000000
    interval = steps if args.smoke else 5000
    packages = {n: importlib.metadata.version(n) for n in (
        "gymnasium","gymnasium-robotics","mujoco","stable-baselines3","jax","flax","optax","torch","numpy","wandb")}
    config = dict(method=args.method, seed=args.seed, smoke=args.smoke, source_commit=commit,
        source_root=str(root), env_id=ENV_ID, env_kwargs=ENV_KWARGS, observation_keys=OBS_KEYS,
        observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
        steps=steps, warmup=warmup, batch_size=256, utd=1, eval_interval=interval,
        eval_episodes=2 if args.smoke else 10, evaluation_modes=agent.modes,
        native=agent.config, packages=packages,
        protocol="online sparse, no HER/reward shaping/offline data; true termination masks; timeout bootstrapping")
    atomic_json(folder/"config.json", config)
    import wandb
    run = wandb.init(project="gmm-trg", entity="OptiQ", name=f"antmaze-umaze-{args.method}-s{args.seed}",
        group="antmaze-umaze-online-20260921", job_type="online-antmaze", config=config,
        dir=str(folder), mode="disabled" if args.smoke else "online", tags=["antmaze",args.method])
    if not args.smoke:
        atomic_json(folder/"wandb.json",dict(id=run.id,url=run.url))
        run.define_metric("env_steps"); run.define_metric("*",step_metric="env_steps")
    evaluator = Evaluator(folder,args.seed,config["eval_episodes"])
    start = time.monotonic(); last_info = {}
    def log(step, values):
        record = dict(env_steps=step, updates=int(agent.updates), elapsed_seconds=time.monotonic()-start, **values)
        if not all(np.isfinite(v) for v in record.values()): raise FloatingPointError(str(record))
        atomic_json(folder/"progress.json",record)
        run.log(record)
        print(json.dumps(record),flush=True)
    def evaluate(step):
        metrics = evaluator.evaluate(agent,step,agent.modes)
        log(step,metrics)
    def save(step):
        audit = parameter_audit(agent)
        directory=folder/"checkpoints"; directory.mkdir(exist_ok=True)
        path=directory/f"agent_{step}.bin"
        agent.save(path)
        files = list(directory.glob(f"*{step}*"))
        if not files: raise RuntimeError("Checkpoint not written")
        atomic_json(folder/"checkpoint.json",dict(step=step,parameter_audit=audit,files=[dict(name=p.name,size=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
    try:
        if not args.smoke: evaluate(0)
        if args.method in ("optiq","sac"):
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.logger import configure
            class Callback(BaseCallback):
                def _on_step(self):
                    step=self.num_timesteps
                    # The final evaluation runs after learn(), including its last update.
                    if step < steps and step % interval == 0: evaluate(step)
                    elif step % 1000 == 0: log(step,{})
                    if step < steps and step % 100000 == 0: save(step)
                    return True
            logger=configure(str(folder/"train_log"),["csv"])
            agent.model.set_logger(logger)
            agent.model.learn(total_timesteps=steps,callback=Callback(),log_interval=1,progress_bar=False)
        else:
            obs,_=env.reset(seed=args.seed)
            for step in range(1,steps+1):
                action=env.action_space.sample() if step <= warmup else agent.act(obs[None])[0]
                if not np.isfinite(action).all(): raise FloatingPointError("Nonfinite training action")
                action=np.clip(action,-1,1).astype(np.float32)
                nxt,reward,terminated,truncated,info=env.step(action)
                agent.store(obs,action,reward,nxt,terminated)
                if step>warmup:
                    last_info=agent.update()
                    if not all(np.isfinite(v) for v in last_info.values()): raise FloatingPointError(str(last_info))
                obs=env.reset()[0] if terminated or truncated else nxt
                if step < steps and step%interval==0: evaluate(step)
                elif step%1000==0: log(step,{f"train/{k}":v for k,v in last_info.items()})
                if step<steps and step%100000==0: save(step)
        evaluate(steps); save(steps)
        final_parameters=parameter_audit(agent)
        if any(final_parameters[k]["sha256"]==v["sha256"] for k,v in initial_parameters.items()):
            raise RuntimeError("Actor or learned critic did not change")
        atomic_json(folder/"parameter-audit.json",dict(initial=initial_parameters,final=final_parameters))
        expected=steps-warmup
        if int(agent.updates)!=expected: raise RuntimeError(f"Update audit {agent.updates} != {expected}")
        primary=agent.modes[0]
        history=evaluator.histories[primary]
        ix=np.asarray(history["timesteps"]) > (steps-100000 if not args.smoke else 0)
        if not args.smoke and ix.sum()!=20: raise RuntimeError("Last 100k must have 20 evaluations")
        result=dict(completed=True,source_commit=commit,method=args.method,seed=args.seed,steps=steps,
            updates=int(agent.updates),primary_mode=primary,
            last100k_success_rate=float(np.mean(np.asarray(history["successes"])[ix])),
            last100k_return=float(np.mean(np.asarray(history["results"])[ix])),
            last100k_min_distance=float(np.mean(np.asarray(history["min_distances"])[ix])),
            seconds=time.monotonic()-start)
        atomic_json(folder/"result.json",result); run.summary.update(result)
    except BaseException as e:
        atomic_json(folder/"failure.json",dict(type=type(e).__name__,message=str(e),source_commit=commit))
        raise
    finally:
        evaluator.close(); env.close(); run.finish()


if __name__=="__main__": main()
