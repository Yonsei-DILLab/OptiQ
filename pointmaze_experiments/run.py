import argparse,importlib.util,json,subprocess,time,random
from pathlib import Path
import numpy as np
from .env import PointMaze,test_geometry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--temperature',type=float,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);test_geometry()
    import jax,torch,flax.serialization as fs
    from omegaconf import OmegaConf
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.buffers import ReplayBuffer
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('point_trg',root/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=module.compose_config(['benchmark=ant','seed=0',f'alg.actor.temperature={args.temperature}','dacer.enabled=false',f'output_root={out}'])
    cfg.env_name='PointMaze-v1-diagnostic';cfg.alg.batch_size=256
    cfg.alg.learning_starts=10000;cfg.alg.actor.learning_starts=10000
    env=PointMaze();budget=10004 if args.smoke else 250000
    model=module.runner.OptiQDIME('MlpPolicy',env=env,cfg=cfg,model_save_path=None,save_every_n_steps=budget)
    model.set_logger(configure(str(out/'learner'),['csv']));model._total_timesteps=budget
    replay=ReplayBuffer(1000000,env.observation_space,env.action_space,device='cpu',n_envs=1)
    model.replay_buffer=replay;p=model.policy
    config=dict(temperature=args.temperature,seed=0,budget=budget,warmup=10000,batch=256,utd=1,N=64,M=64,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reward='negative Euclidean distance',observation='xy',action='xy displacement, 0.2 per axis per step',
        train_start='uniform[-2,2]^2',eval_start=[0,0],horizon=500,goal=[-8,0],goal_radius=.5,
        algorithm=OmegaConf.to_container(cfg,resolve=True))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    run=None
    if not args.smoke:
        import wandb
        run=wandb.init(entity='OptiQ',project='jaehun-pointmaze',name=f'iBOLT-PointMaze-v1-T{args.temperature:g}-s0',config=config,dir=str(out))
    def log(d,step):
        row=dict(env_steps=step,**d)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if run:run.log(row,step=step)
    def act(obs,noise):
        p.reset_noise()
        return np.asarray(p.sample_action(p.actor_state,obs,p.noise_key,deterministic=False,sample_conditional_noise=noise)).copy()
    def evaluate(step,n):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle,Circle
        saved=p.key,p.noise_key,np.random.get_state(),random.getstate()
        p.key=jax.random.PRNGKey(42000)
        try:
            actions=act(np.zeros((4096,2),np.float32),False)
            np.savez_compressed(out/f's0-actions-{step:07d}.npz',actions=actions)
            envs=[PointMaze(random_start=False) for _ in range(n)]
            obs=np.stack([e.reset()[0] for e in envs]);paths=[[o.copy()] for o in obs]
            rets=np.zeros(n);success=np.zeros(n,bool);active=np.ones(n,bool)
            for _ in range(500):
                aa=act(obs,False)
                for i in np.flatnonzero(active):
                    obs[i],r,term,trunc,info=envs[i].step(aa[i]);rets[i]+=r;paths[i].append(obs[i].copy())
                    if term or trunc:active[i]=False;success[i]=info['success']
                if not active.any():break
            xy=np.full((n,501,2),np.nan);routes=[]
            fig,ax=plt.subplots(figsize=(6,5))
            ax.add_patch(Rectangle((-6,-2),4,4,color='lightgray'));ax.add_patch(Circle((-8,0),.5,color='green'))
            for i,path in enumerate(paths):
                path=np.array(path);xy[i,:len(path)]=path
                hit=np.flatnonzero(path[:,0]<=-4);routes.append(0 if not len(hit) else int(np.sign(path[hit[0],1])))
                ax.plot(path[:,0],path[:,1],alpha=.25,lw=.7)
            ax.scatter(0,0,color='black');ax.set(xlim=(-10,2),ylim=(-6,6),aspect='equal',title=f'T={args.temperature}, step={step}, random z / mu only')
            fig.savefig(out/f'rollouts-{step:07d}.png',dpi=160);plt.close(fig)
            np.savez_compressed(out/f'rollouts-{step:07d}.npz',xy=xy,returns=rets,success=success,routes=routes)
            log(dict(eval_return=float(rets.mean()),success=float(success.mean()),upper=routes.count(1),lower=routes.count(-1),s0_action_std=actions.std(axis=0).tolist()),step)
            if run:run.log({'rollouts':wandb.Image(str(out/f'rollouts-{step:07d}.png'))},step=step)
        finally:p.key,p.noise_key=saved[:2];np.random.set_state(saved[2]);random.setstate(saved[3])
    obs,_=env.reset();start=time.time();updates=0
    for step in range(1,budget+1):
        action=env.action_space.sample() if step<=10000 else act(obs[None],True)[0]
        nxt,reward,term,trunc,info=env.step(action)
        replay.add(obs[None],nxt[None],action[None],np.array([reward]),np.array([term or trunc]),[{'TimeLimit.truncated':trunc}])
        obs=env.reset()[0] if term or trunc else nxt
        if step>10000:
            model.num_timesteps=step;model._current_progress_remaining=1-step/budget
            model.train(batch_size=256,gradient_steps=1);updates+=1
            if step%1000==0 or args.smoke:
                metrics={k:float(v) for k,v in model.logger.name_to_value.items() if isinstance(v,(int,float,np.number))}
                assert all(np.isfinite(v) for v in metrics.values()),metrics
                log(dict(updates=updates,elapsed=time.time()-start,**metrics),step)
        if (not args.smoke and step%5000==0) or (args.smoke and step==budget):
            state=dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state)
            data=fs.to_bytes(state);fs.from_bytes(state,data)
            (out/f'policy-{step:07d}.msgpack').write_bytes(data)
            evaluate(step,4 if args.smoke else 100)
    assert updates==budget-10000
    log(dict(completed=True,updates=updates,smoke=args.smoke),budget)
    if run:run.finish()
if __name__=='__main__':main()
