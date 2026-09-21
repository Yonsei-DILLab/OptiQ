"""Online learned-policy path diversity with an explicit interaction budget."""
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
from antmaze.evaluation import atomic_json
from .env import make_env,geometry
from .coverage import Coverage
from .evaluation import Evaluator


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--method",required=True,choices=["optiq","sac","meow","sql","mfpo","dipo"])
    p.add_argument("--task",required=True,choices=["v1","v3","v4"])
    p.add_argument("--seed",type=int,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--campaign",default="antmaze-multimodal-100k-20260921")
    p.add_argument("--eval-interval",type=int,default=5000)
    p.add_argument("--checkpoint-interval",type=int,default=50000)
    a=p.parse_args();cpus=sorted(os.sched_getaffinity(0));gpu=int(os.environ.get("CAMPAIGN_GPU","0"))
    os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    import torch
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.backends.cudnn.deterministic=True
    random.seed(a.seed);np.random.seed(a.seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if a.method in ("optiq","mfpo","sql"):
        import jax
        if jax.default_backend()!="gpu":raise RuntimeError("JAX GPU required")
    from antmaze.agents import DIPO,MEOW,MFPO,SQL,SB3,parameter_audit
    root=Path(__file__).resolve().parents[2];folder=a.output;folder.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip():
        raise RuntimeError("Commit source before training")
    env=Coverage(make_env(a.task,a.seed))
    if a.method in ("optiq","sac"):
        agent=SB3(a.method,env,a.seed,folder,a.smoke)
        if a.method=="optiq":
            agent.model.cfg.env_name=f"DDiffPG-{a.task}-dense-port"
            agent.config["env_name"]=agent.model.cfg.env_name
    else:agent=dict(dipo=DIPO,meow=MEOW,mfpo=MFPO,sql=SQL)[a.method](env,a.seed,folder)
    initial=parameter_audit(agent);warmup=256 if a.smoke else agent.warmup
    steps=272 if a.smoke else a.steps;interval=steps if a.smoke else a.eval_interval
    if interval<=0 or a.checkpoint_interval<=0:raise ValueError("Intervals must be positive")
    if steps<=warmup:raise ValueError("Interaction budget must exceed warmup")
    config=dict(method=a.method,task=a.task,seed=a.seed,smoke=a.smoke,source_commit=commit,
        source_root=str(root),steps=steps,campaign=a.campaign,warmup=warmup,batch_size=256,utd=1,native=agent.config,
        environment=geometry(a.task),xml_sha256=env.unwrapped.xml_sha256,
        observation="qpos[:15]+qvel[:14], xy included, no specified-goal conditioning",
        reward="-min distance to any goal; no success bonus/locomotion/control/survival rewards",
        reset="v1 xy uniform[-2,2]^2; v3/v4 fixed origin; default robot pose and zero velocity",
        evaluation="direct learned-policy sampling; no mean/best-of-K/external DACER noise",
        trajectory_episodes_per_reset_mode=100,trajectory_reset_modes=["natural","fixed_full_simulator_state"],
        interval_eval_episodes=10,eval_interval=interval,checkpoint_interval=a.checkpoint_interval,
        packages={n:importlib.metadata.version(n) for n in ("mujoco","gymnasium","jax","flax","optax","torch","numpy","wandb")})
    atomic_json(folder/"config.json",config)
    import wandb
    run=wandb.init(project="gmm-trg",entity="OptiQ",name=f"antmaze-{a.task}-{a.method}-s{a.seed}-{steps//1000}k",
        group=f"{a.campaign}-{a.task}",job_type="multimodal-policy",
        config=config,dir=str(folder),mode="disabled" if a.smoke else "online",tags=["antmaze","path-diversity",a.method,a.task])
    if not a.smoke:
        atomic_json(folder/"wandb.json",dict(id=run.id,url=run.url))
        run.define_metric("env_steps");run.define_metric("*",step_metric="env_steps")
    batch=2 if a.smoke else 10;evaluator=Evaluator(folder,a.task,a.seed,batch)
    started=time.monotonic();last_info={}
    def log(step,data):
        training={k:v for k,v in env.training_summary().items() if v is not None}
        record=dict(env_steps=step,updates=int(agent.updates),seconds=time.monotonic()-started,**training,**data)
        if not all(np.isfinite(v) for v in record.values()):raise FloatingPointError(str(record))
        atomic_json(folder/"progress.json",record);run.log(record);print(json.dumps(record),flush=True)
    def evaluate(step):
        s=evaluator.evaluate(agent,step,episodes=batch)
        log(step,{f"eval/policy/{k}":s[k] for k in ("success_rate","mean_return","mean_min_distance","mean_final_distance")})
        env.save(folder/"training_coverage.npz")
    def save(step):
        audit=parameter_audit(agent);directory=folder/"checkpoints";directory.mkdir(exist_ok=True)
        agent.save(directory/f"agent_{step}.bin")
        files=list(directory.glob(f"*{step}*"))
        if not files:raise RuntimeError("Missing checkpoint")
        atomic_json(folder/"checkpoint.json",dict(step=step,parameter_audit=audit,
            files=[dict(name=p.name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
    try:
        if not a.smoke:evaluate(0)
        if a.method in ("optiq","sac"):
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.logger import configure
            class Callback(BaseCallback):
                def _on_step(self):
                    step=self.num_timesteps
                    if step<steps and step%interval==0:evaluate(step)
                    elif step%1000==0:log(step,{})
                    if step<steps and step%a.checkpoint_interval==0:save(step)
                    return True
            agent.model.set_logger(configure(str(folder/"train_log"),["csv"]))
            agent.model.learn(total_timesteps=steps,callback=Callback(),progress_bar=False,log_interval=1)
        else:
            obs,_=env.reset(seed=a.seed)
            for step in range(1,steps+1):
                action=env.action_space.sample() if step<=warmup else agent.act(obs[None])[0]
                if not np.isfinite(action).all():raise FloatingPointError("Nonfinite training action")
                action=np.clip(action,-1,1).astype(np.float32)
                nxt,r,t,tr,info=env.step(action);agent.store(obs,action,r,nxt,t)
                if step>warmup:
                    last_info=agent.update()
                    if not all(np.isfinite(v) for v in last_info.values()):raise FloatingPointError(str(last_info))
                obs=env.reset()[0] if t or tr else nxt
                if step<steps and step%interval==0:evaluate(step)
                elif step%1000==0:log(step,{f"train/{k}":v for k,v in last_info.items()})
                if step<steps and step%a.checkpoint_interval==0:save(step)
        evaluate(steps);save(steps)
        final=parameter_audit(agent)
        if int(agent.updates)!=steps-warmup:raise RuntimeError("Learner update count mismatch")
        if any(final[k]["sha256"]==v["sha256"] for k,v in initial.items()):raise RuntimeError("Actor or critic did not change")
        atomic_json(folder/"parameter-audit.json",dict(initial=initial,final=final))
        summaries={}
        for mode in (["policy","mu_only"] if a.method=="optiq" else ["policy"]):
            for fixed in (False,True):
                label=mode+"-"+("fixed" if fixed else "natural")
                s=evaluator.evaluate(agent,steps,episodes=batch if a.smoke else 100,fixed=fixed,mode=mode,archive=True)
                summaries[label]=s
                run.log({"env_steps":steps,**{f"trajectories/{label}/{k}":s[k] for k in ("success_rate","mean_return")},
                    f"trajectories/{label}/effective_routes":s["successful_routes"]["effective_modes"],
                    f"trajectories/{label}/dominant_route_fraction":s["successful_routes"]["dominant_fraction"]})
        result=dict(completed=True,source_commit=commit,method=a.method,task=a.task,seed=a.seed,
            steps=steps,updates=int(agent.updates),seconds=time.monotonic()-started,summaries=summaries,
            training=env.training_summary())
        atomic_json(folder/"result.json",result)
        run.summary.update(dict(completed=True,steps=steps,updates=int(agent.updates),
            **env.training_summary(),
            fixed_policy_success=summaries["policy-fixed"]["success_rate"],
            fixed_policy_effective_routes=summaries["policy-fixed"]["successful_routes"]["effective_modes"]))
        print(f"Completed {a.task} {a.method} seed{a.seed}",flush=True)
    except BaseException as e:
        atomic_json(folder/"failure.json",dict(type=type(e).__name__,message=str(e),source_commit=commit));raise
    finally:evaluator.close();env.close();run.finish()


if __name__=="__main__":main()
