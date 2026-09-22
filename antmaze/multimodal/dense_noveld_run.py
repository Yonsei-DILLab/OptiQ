"""One-env, UTD1 dense+NovelD experiment with complete continuation state."""
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
from . import resume

CAMPAIGN="antmaze-dense-noveld-1m-s0-20260922"


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--method",required=True,choices=["optiq","sac","meow","mfpo"])
    p.add_argument("--task",required=True,choices=["v1","v2","v3","v4"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1000000)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",type=Path)
    p.add_argument("--noveld-coefficient",type=float,default=.01)
    p.add_argument("--campaign-name",default=CAMPAIGN)
    p.add_argument("--run-name")
    p.add_argument("--profile",default="dense-noveld-1m")
    a=p.parse_args();seed=0
    cpus=sorted(os.sched_getaffinity(0));gpu=int(os.environ.get("CAMPAIGN_GPU","0"))
    os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    import torch
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cudnn.deterministic=True
    random.seed(seed);np.random.seed(seed)
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
    env=Coverage(make_env(a.task,seed));obs,_=env.reset(seed=seed)
    agent=SB3(a.method,env,seed,folder,smoke=a.smoke) if a.method in ("optiq","sac") else dict(meow=MEOW,mfpo=MFPO)[a.method](env,seed,folder)
    warmup=256 if a.smoke else agent.warmup
    steps=a.steps;assert steps>warmup
    intrinsic=NovelD(29,seed,coefficient=a.noveld_coefficient)
    replay=Replay(1000000,seed,intrinsic,device="cpu" if a.method=="optiq" else "cuda",dictionary=a.method=="mfpo")
    if isinstance(agent,SB3):
        from stable_baselines3.common.logger import configure
        agent.model.replay_buffer=replay
        agent.model.set_logger(configure(str(folder/"train_log"),["csv"]))
        agent.model._total_timesteps=steps;agent.model._current_progress_remaining=1.
        if a.method=="optiq":
            agent.model.cfg.env_name=f"DDiffPG-{a.task}-dense-noveld-port"
            agent.config["env_name"]=agent.model.cfg.env_name
    else:agent.buffer=replay
    evaluator=Evaluator(folder,a.task,seed,batch=2 if a.smoke else 10)
    behavior_rng=np.random.default_rng(seed+40821)
    def rnd_hash(net):return resume.digest(net.state_dict())
    initial=parameter_audit(agent)
    initial_rnd=dict(predictor=rnd_hash(intrinsic.predictor),target=rnd_hash(intrinsic.target))
    config=dict(profile=a.profile,method=a.method,task=a.task,seed=seed,smoke=a.smoke,
        source_commit=commit,source_root=str(root),steps=steps,warmup=warmup,batch_size=256,
        num_envs=1,utd=1,expected_updates=steps-warmup,native=agent.config,intrinsic=intrinsic.config,
        environment=geometry(a.task),xml_sha256=env.unwrapped.xml_sha256,
        reward="environment=-nearest goal distance; learner adds NovelD; evaluation excludes NovelD",
        reward_provenance="MaxEntDP D.2 dense distance penalty + user-requested DDiffPG NovelD; not exact unpublished MaxEntDP AntMaze reproduction",
        eval_modes=dict(native="SAC tanh(mu); MFPO Q-best-of10; MEOW center-prior; OptiQ random-z mu-only",
            policy="direct stochastic policy including conditional sigma for OptiQ; no external DACER noise",
            zero_z="OptiQ z=0, mu-only"),
        eval_interval=25000,checkpoint_interval=100000,trajectory_episodes_per_reset_mode=100,
        resume_from=str(a.resume) if a.resume else None,
        packages={n:importlib.metadata.version(n) for n in ("mujoco","gymnasium","jax","flax","optax","torch","numpy","wandb")})
    if a.resume:
        extra,proof=resume.load(a.resume,agent,replay,env,behavior_rng,evaluator,commit,a.task,a.method)
        assert extra["warmup"]==warmup and extra["smoke"]==a.smoke
        initial=extra["initial"];initial_rnd=extra["initial_rnd"];obs=env.unwrapped.observation()
        assert env.total_steps<steps
        atomic_json(folder/"resume-verification.json",dict(passed=True,loaded_step=env.total_steps,
            loaded_updates=agent.updates,proof=proof,source=str(a.resume),
            includes=["replay","all_optimizers","actor_and_target","critic_and_target","entropy","DACER","NovelD","RNG","simulator"]))
    atomic_json(folder/"config.json",config)
    # Creating the tracking client must not alter the restored training RNG.
    from antmaze.evaluation import isolated_rng
    import wandb
    with isolated_rng(18372):
        run=wandb.init(project="gmm-trg",entity="OptiQ",name=a.run_name or f"antmaze-{a.task}-{a.method}-dense-noveld-s0-1m",
            group=a.campaign_name,job_type="multimodal-policy",config=config,dir=str(folder),
            mode="disabled" if a.smoke else "online",tags=["antmaze","noveld","dense",a.task,a.method,"utd1"])
    if not a.smoke:
        atomic_json(folder/"wandb.json",dict(id=run.id,url=run.url))
        run.define_metric("env_steps");run.define_metric("*",step_metric="env_steps")
    started=time.monotonic();step=env.total_steps
    timing=dict(collection=0.,learner=0.,evaluation=0.,checkpoint=0.)
    def log():
        record=dict(env_steps=step,updates=int(agent.updates),seconds=time.monotonic()-started,
            **env.training_summary(),**intrinsic.metrics,**{f"seconds/{k}":v for k,v in timing.items()})
        atomic_json(folder/"progress.json",record);run.log(record);print(json.dumps(record),flush=True)
        if isinstance(agent,SB3):agent.model.logger.dump(step)
    def evaluate():
        t=time.monotonic()
        for mode in ("native","policy"):
            s=evaluator.evaluate(agent,step,episodes=2 if a.smoke else 10,mode=mode)
            run.log(dict(env_steps=step,**{f"eval/{mode}/{k}":s[k] for k in ("success_rate","mean_return","mean_min_distance")}))
        timing["evaluation"]+=time.monotonic()-t;env.save(folder/"training_coverage.npz")
    def save():
        t=time.monotonic()
        extra=dict(source_commit=commit,method=a.method,task=a.task,seed=seed,warmup=warmup,
                   smoke=a.smoke,initial=initial,initial_rnd=initial_rnd)
        proof=resume.save(folder/"resume"/f"step_{step:010d}",agent,replay,env,behavior_rng,evaluator,extra)
        timing["checkpoint"]+=time.monotonic()-t
        atomic_json(folder/"checkpoint.json",proof)
    def action():
        if step<warmup:return behavior_rng.uniform(-1,1,8).astype(np.float32)
        if a.method=="optiq":
            policy=agent.model.policy;policy.reset_noise()
            x=np.asarray(policy.sample_action(policy.actor_state,obs[None],policy.noise_key,
                deterministic=False,sample_conditional_noise=True))[0]
            x=x+agent.model.regulator_noise_std*agent.model.regulator_rng.normal(size=x.shape)
        elif a.method=="sac":x=agent.model.predict(obs,deterministic=False)[0]
        else:x=agent.act(obs[None])[0]
        assert x.shape==(8,) and np.isfinite(x).all()
        return np.clip(x,-1,1).astype(np.float32)
    def update():
        if isinstance(agent,SB3):
            m=agent.model
            replay.diagnostic=bool(a.method=="optiq" and m.regulator_enabled and m._n_updates>=m.regulator_next_update)
            m.train(batch_size=256,gradient_steps=1)
        else:
            info=agent.update()
            if not all(np.isfinite(float(v)) for v in info.values()):raise FloatingPointError(info)
    try:
        if not a.smoke and not a.resume:evaluate()
        while step<steps:
            t=time.monotonic();act=action()
            next_obs,r,term,trunc,info=env.step(act)
            assert r==-info["distance"]
            replay.add_batch(obs[None],act[None],np.array([r]),next_obs[None],np.array([term]))
            obs=env.reset()[0] if term or trunc else next_obs
            step+=1;timing["collection"]+=time.monotonic()-t
            if isinstance(agent,SB3):
                agent.model.num_timesteps=step;agent.model._current_progress_remaining=1-step/steps
            if step>warmup:
                t=time.monotonic();update();timing["learner"]+=time.monotonic()-t
            if not a.smoke and step%25000==0 and step<steps:evaluate()
            if step%100000==0 and step<steps:save()
            if step%1000==0 or step==steps:log()
        assert agent.updates==intrinsic.updates==steps-warmup
        evaluate();save()
        final=parameter_audit(agent)
        assert all(final[k]["sha256"]!=v["sha256"] for k,v in initial.items())
        final_rnd=dict(predictor=rnd_hash(intrinsic.predictor),target=rnd_hash(intrinsic.target))
        assert initial_rnd["target"]==final_rnd["target"] and initial_rnd["predictor"]!=final_rnd["predictor"]
        atomic_json(folder/"parameter-audit.json",dict(initial=initial,final=final))
        atomic_json(folder/"intrinsic-audit.json",dict(initial=initial_rnd,final=final_rnd,updates=intrinsic.updates,passed=True))
        summaries={}
        for mode in (["native","policy","zero_z"] if a.method=="optiq" else ["native","policy"]):
            for fixed in (False,True):
                label=mode+"-"+("fixed" if fixed else "natural")
                s=evaluator.evaluate(agent,step,episodes=2 if a.smoke else 100,fixed=fixed,mode=mode,archive=True)
                summaries[label]=s
                run.log({"env_steps":step,**{f"trajectories/{label}/{k}":s[k] for k in ("success_rate","mean_return")}})
        result=dict(completed=True,source_commit=commit,method=a.method,task=a.task,seed=seed,
            steps=step,updates=int(agent.updates),seconds=time.monotonic()-started,summaries=summaries,
            timing=timing,training=env.training_summary(),replay_size=replay.size)
        atomic_json(folder/"result.json",result);run.summary.update(dict(completed=True,steps=step,updates=agent.updates))
        print("COMPLETED "+a.method,flush=True)
    except BaseException as error:
        atomic_json(folder/"failure.json",dict(type=type(error).__name__,message=str(error),source_commit=commit));raise
    finally:evaluator.close();env.close();run.finish()


if __name__=="__main__":main()
