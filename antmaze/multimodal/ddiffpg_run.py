"""Dense AntMaze + DDiffPG NovelD/collection schedule; native four learners."""
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
from .evaluation import Evaluator
from .vector_env import Collector

PROFILE="ddiffpg-dense-noveld"
CAMPAIGN="antmaze-v1-noveld-100k-s0-20260921"


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--method",required=True,choices=["optiq","sac","mfpo","meow"])
    p.add_argument("--task",default="v1",choices=["v1"])
    p.add_argument("--seed",type=int,default=0,choices=[0])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    cpus=sorted(os.sched_getaffinity(0));gpu=int(os.environ.get("CAMPAIGN_GPU","0"))
    os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    import torch
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.backends.cudnn.deterministic=True
    random.seed(a.seed);np.random.seed(a.seed)
    assert torch.cuda.is_available()
    if a.method in ("optiq","mfpo"):
        import jax
        assert jax.default_backend()=="gpu"
    from antmaze.agents import SB3,MEOW,MFPO,parameter_audit
    from .noveld import NovelD,Replay
    folder=a.output;folder.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip()
    env=make_env(a.task,a.seed)
    agent=SB3(a.method,env,a.seed,folder) if a.method in ("optiq","sac") else dict(meow=MEOW,mfpo=MFPO)[a.method](env,a.seed,folder)
    steps=8704 if a.smoke else 100000
    warmup=8192;batch_size=4096;num_envs=256;updates_per_round=8
    expected_updates=int(np.ceil((steps-warmup)/num_envs))*updates_per_round
    intrinsic=NovelD(29,a.seed)
    replay=Replay(1000000,a.seed,intrinsic,device="cpu" if a.method=="optiq" else "cuda",dictionary=a.method=="mfpo")
    if isinstance(agent,SB3):
        from stable_baselines3.common.logger import configure
        agent.model.replay_buffer=replay
        agent.model.set_logger(configure(str(folder/"train_log"),["csv"]))
        agent.model._total_timesteps=steps
        agent.model.learning_starts=warmup
        agent.model.batch_size=batch_size
        agent.model.gradient_steps=updates_per_round
        if a.method=="optiq":
            agent.model.cfg.env_name="DDiffPG-v1-dense-noveld-port"
            agent.model.cfg.alg.actor.learning_starts=warmup
            agent.config["env_name"]=agent.model.cfg.env_name
    else:agent.buffer=replay
    initial=parameter_audit(agent)
    def rnd_hash(model):
        return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
    initial_rnd=dict(predictor=rnd_hash(intrinsic.predictor),target=rnd_hash(intrinsic.target))
    config=dict(profile=PROFILE,method=a.method,task=a.task,seed=a.seed,smoke=a.smoke,
        source_commit=commit,source_root=str(root),steps=steps,warmup=warmup,batch_size=batch_size,
        num_envs=num_envs,updates_per_round=updates_per_round,utd=updates_per_round/num_envs,
        expected_updates=expected_updates,native=agent.config,intrinsic=intrinsic.config,
        overrides="NovelD + 256-env collection, batch4096, 8 updates/round,8192 warmup; native networks/LR/tau retained",
        environment=geometry(a.task),xml_sha256=env.xml_sha256,
        reward="environment=-nearest goal distance; training adds NovelD; evaluation excludes NovelD",
        collection_workers=8,dacer_entropy_batch_size=256,
        eval_interval=25000,checkpoint_interval=50000,trajectory_episodes_per_reset_mode=100,
        packages={n:importlib.metadata.version(n) for n in ("mujoco","gymnasium","jax","flax","optax","torch","numpy","wandb")})
    atomic_json(folder/"config.json",config)
    import wandb
    run=wandb.init(project="gmm-trg",entity="OptiQ",name=f"antmaze-v1-{a.method}-noveld-s0-100k",
        group=CAMPAIGN,job_type="multimodal-policy",config=config,dir=str(folder),
        mode="disabled" if a.smoke else "online",tags=["antmaze","noveld","dense","v1",a.method])
    if not a.smoke:
        atomic_json(folder/"wandb.json",dict(id=run.id,url=run.url))
        run.define_metric("env_steps");run.define_metric("*",step_metric="env_steps")
    collector=Collector(a.task,a.seed,num_envs,workers=8)
    evaluator=Evaluator(folder,a.task,a.seed,batch=2 if a.smoke else 10)
    behavior_rng=np.random.default_rng(a.seed+40821)
    started=time.monotonic();timing=dict(collection=0.,learner=0.,evaluation=0.)
    def log(step):
        record=dict(env_steps=step,updates=int(agent.updates),seconds=time.monotonic()-started,
            training_successes=int(collector.successes),training_episodes=int(collector.episodes),
            **intrinsic.metrics,**{f"seconds/{k}":v for k,v in timing.items()})
        assert all(np.isfinite(v) for v in record.values())
        atomic_json(folder/"progress.json",record);run.log(record);print(json.dumps(record),flush=True)
    def evaluate(step):
        t=time.monotonic()
        result=evaluator.evaluate(agent,step,episodes=2 if a.smoke else 10)
        timing["evaluation"]+=time.monotonic()-t
        run.log(dict(env_steps=step,**{"eval/"+k:result[k] for k in ("success_rate","mean_return","mean_min_distance")}))
        collector.save(folder/"training_coverage.npz")
    def save(step):
        directory=folder/"checkpoints";directory.mkdir(exist_ok=True)
        agent.save(directory/f"agent_{step}.bin");intrinsic.save(directory/f"noveld_{step}.pt")
        files=list(directory.glob(f"*{step}*"))
        atomic_json(folder/"checkpoint.json",dict(step=step,parameter_audit=parameter_audit(agent),
            files=[dict(name=f.name,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]))
    def actions(obs):
        if a.method=="optiq":
            policy=agent.model.policy;policy.reset_noise()
            x=np.asarray(policy.sample_action(policy.actor_state,obs,policy.noise_key,
                deterministic=False,sample_conditional_noise=True))
            # Native DACER behavior noise, independent of policy evaluation.
            x=x+agent.model.regulator_noise_std*agent.model.regulator_rng.normal(size=x.shape)
        elif a.method=="sac":x=agent.model.predict(obs,deterministic=False)[0]
        elif a.method=="mfpo":
            x,agent.agent=agent.agent.eval_actions_sample_batch(obs);x=np.asarray(x)
        else:x=agent.act(obs)
        assert x.shape==(len(obs),8) and np.isfinite(x).all()
        return np.clip(x,-1,1).astype(np.float32)
    def update():
        if isinstance(agent,SB3):
            model=agent.model
            replay.diagnostic=bool(a.method=="optiq" and model.regulator_enabled and model._n_updates>=model.regulator_next_update)
            model.train(batch_size=batch_size,gradient_steps=1)
        elif a.method=="mfpo":
            agent.agent,info=agent.agent.update(replay.sample(batch_size),utd_ratio=1)
            if not all(np.isfinite(float(v)) for v in info.values()):raise FloatingPointError(info)
            agent.updates+=1
        else:
            from torch.nn import functional as F
            d=replay.sample(batch_size)
            with torch.no_grad():
                agent.target.eval();v=agent.target.get_v(torch.cat([d.next_observations]*2))
                target=d.rewards+(1-d.dones)*.99*torch.minimum(v[:batch_size],v[batch_size:])
            agent.policy.train()
            q,_=agent.policy.get_qv(torch.cat([d.observations]*2),torch.cat([d.actions]*2))
            loss=F.mse_loss(q,torch.cat([target]*2));assert torch.isfinite(loss)
            agent.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(),30.);agent.optimizer.step()
            with torch.no_grad():
                for p,t in zip(agent.policy.parameters(),agent.target.parameters()):t.lerp_(p,.005)
            agent.updates+=1
    try:
        if not a.smoke:evaluate(0)
        step=0;next_eval=25000;next_checkpoint=50000
        while step<steps:
            n=min(num_envs,steps-step);t=time.monotonic()
            action=behavior_rng.uniform(-1,1,(n,8)).astype(np.float32) if step<warmup else actions(collector.obs[:n])
            replay.add_batch(*collector.step(action));step+=n
            timing["collection"]+=time.monotonic()-t
            if isinstance(agent,SB3):
                agent.model.num_timesteps=step
                agent.model._current_progress_remaining=1-step/steps
            if step>warmup:
                t=time.monotonic()
                for _ in range(updates_per_round):update()
                timing["learner"]+=time.monotonic()-t
            if step>=next_eval and step<steps:
                evaluate(step);next_eval+=25000
            if step>=next_checkpoint and step<steps:
                save(step);next_checkpoint+=50000
            if step%2048==0 or step==steps:log(step)
        assert agent.updates==intrinsic.updates==expected_updates
        evaluate(steps);save(steps)
        final=parameter_audit(agent)
        assert all(final[k]["sha256"]!=v["sha256"] for k,v in initial.items())
        final_rnd=dict(predictor=rnd_hash(intrinsic.predictor),target=rnd_hash(intrinsic.target))
        assert initial_rnd["target"]==final_rnd["target"] and initial_rnd["predictor"]!=final_rnd["predictor"]
        atomic_json(folder/"parameter-audit.json",dict(initial=initial,final=final))
        atomic_json(folder/"intrinsic-audit.json",dict(initial=initial_rnd,final=final_rnd,updates=intrinsic.updates,
            expected_updates=expected_updates,metrics=intrinsic.metrics,passed=True))
        summaries={}
        for mode in (["policy","mu_only"] if a.method=="optiq" else ["policy"]):
            for fixed in (False,True):
                label=mode+"-"+("fixed" if fixed else "natural")
                s=evaluator.evaluate(agent,steps,episodes=2 if a.smoke else 100,fixed=fixed,mode=mode,archive=True)
                summaries[label]=s
                run.log({"env_steps":steps,**{f"trajectories/{label}/{k}":s[k] for k in ("success_rate","mean_return")}})
        result=dict(completed=True,profile=PROFILE,source_commit=commit,method=a.method,task=a.task,seed=a.seed,
            steps=steps,updates=int(agent.updates),seconds=time.monotonic()-started,summaries=summaries,timing=timing)
        atomic_json(folder/"result.json",result);run.summary.update(dict(completed=True,steps=steps,updates=agent.updates))
        print("COMPLETED "+a.method,flush=True)
    except BaseException as error:
        atomic_json(folder/"failure.json",dict(type=type(error).__name__,message=str(error),source_commit=commit));raise
    finally:collector.close();evaluator.close();env.close();run.finish()


if __name__=="__main__":main()
