"""Real online actor-critic training, separate from fixed-Q reconstruction."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
from .target import ROOT,RESULTS,initialize_target
from .evaluation import atomic_json
from .navigation import GMM40Navigation
from .navigation_evaluation import evaluate_navigation


@contextmanager
def evaluation_rng():
    import torch
    devices=[0] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(92001)
        if devices:torch.cuda.manual_seed_all(92001)
        yield


def run_navigation(args):
    if args.method in ('optiq','direct_gmm'):
        if args.n <= 0 or args.m <= 0 or args.m % args.n:
            raise ValueError('OptiQ requires positive N and M divisible by N')
        if args.epsilon <= 0 or args.sinkhorn_iterations <= 0:
            raise ValueError('OptiQ requires positive OT epsilon and iterations')
    initialize_target()
    folder=RESULTS/args.name;folder.mkdir(parents=True,exist_ok=False)
    (folder/"checkpoints").mkdir()
    config=dict(method=args.method,seed=args.seed,actor_updates=args.steps,warmup=args.warmup,
                total_env_steps=args.steps+args.warmup,temperature=(args.temperature if args.method in ('optiq','direct_gmm','meow') else 'automatic' if args.method in ('sac','mfpo') else None),gamma=.99,batch_size=256,
                state_space=[-50,50],start="uniform [-50,50]^2",movement="unit-normalized action",horizon=100,
                reward="DiKL original GMM40 log density at next physical position",Q="learned, not oracle",
                eval_episodes=args.eval_episodes,eval_rng_isolated=True,upstream_reference="DQS-reference/src/envs/gmm_env.py",
                observations="raw physical (x,y)",pid=os.getpid())
    config.update(temperature_policy=('fixed tunable teacher temperature' if args.method in ('optiq','direct_gmm') else 'native'),
                  q_probe_reference=('live twin mean, matching the OptiQ teacher' if args.method in ('optiq','direct_gmm') else 'native conservative live Q'),
                  actor_hidden_dims=([args.width]*args.depth if args.method in ('optiq','direct_gmm','sac') else 'upstream native architecture'),
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    if args.method in ('optiq','direct_gmm'):
        config.update(distillation_loss='direct_gmm_nll' if args.method=='direct_gmm' else 'conditional_ot_nll',
                      ot_enabled=args.method=='optiq',n=args.n,m=args.m,epsilon=args.epsilon,
                      sinkhorn_iterations=args.sinkhorn_iterations)
    if args.method=='mfpo':config['critic_value_support']=dict(min=-1600,max=1600,atoms=101,clipped_mass_diagnostics_every=1000)
    config['source_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    config['learner_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'optiq_dime').glob('*.py'))}
    atomic_json(folder/"config.json",config)
    atomic_json(folder/"status.json",dict(status="initializing",pid=os.getpid(),updates=0))
    try:
        if args.method in ('optiq','direct_gmm','sac'):run_sb3(args,folder)
        else:run_custom(args,folder)
        from .audit_updates import audit
        count_audit=audit(folder)
        if count_audit['errors']:raise RuntimeError(count_audit['errors'])
        if not args.name.startswith('validation_'):
            from .report import refresh
            refresh()
    except Exception as exc:
        atomic_json(folder/"status.json",dict(status="failed",pid=os.getpid(),error=repr(exc)))
        raise


def run_sb3(args,folder):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    env=GMM40Navigation()
    if args.method in ('optiq','direct_gmm'):
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf,open_dict
        from optiq_dime import OptiQDIME
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs')):cfg=compose(config_name='mujoco_v5_direct_gmm' if args.method=='direct_gmm' else 'mujoco_v5')
        with open_dict(cfg):
            cfg.seed=args.seed;cfg.env_name='GMM40Navigation';cfg.task='gmm40';cfg.total_steps=args.steps+args.warmup
            cfg.alg.actor.temperature=args.temperature;cfg.alg.learning_starts=args.warmup;cfg.alg.actor.learning_starts=args.warmup
            cfg.alg.actor.num_policy_samples=args.n
            cfg.alg.actor.proposals_per_policy_sample=args.m//args.n
            cfg.alg.actor.sinkhorn_epsilon=args.epsilon
            cfg.alg.actor.sinkhorn_iterations=args.sinkhorn_iterations
            cfg.alg.actor.hidden_dims=[args.width]*args.depth;cfg.alg.critic.hs=[args.width]*args.depth
            cfg.wandb.activate=False;cfg.diagnostic_interval=1000
        OmegaConf.save(cfg,folder/'resolved_optiq.yaml')
        model=OptiQDIME('MlpPolicy',env,model_save_path=str(folder/'checkpoints'),save_every_n_steps=100000000,cfg=cfg)
    else:
        from stable_baselines3 import SAC
        model=SAC('MlpPolicy',env,learning_rate=3e-4,buffer_size=1000000,learning_starts=args.warmup,
                  batch_size=256,tau=.005,gamma=.99,train_freq=1,gradient_steps=1,ent_coef='auto',
                  policy_kwargs=dict(net_arch=[args.width]*args.depth),seed=args.seed,device='cuda',verbose=0)
    model.set_logger(configure(str(folder/'training_logs'),['csv']))
    from .model_sizes import save_sizes
    if args.method in ('optiq','direct_gmm'):save_sizes(folder,actor=model.policy.actor_state,critic=model.policy.qf_state)
    else:save_sizes(folder,actor=model.actor,critic=model.critic)
    class Callback(BaseCallback):
        def __init__(self):super().__init__();self.last=-1;self.started=time.monotonic()
        def snapshot(self):
            step=self.model._n_updates
            if step==self.last:return
            self.last=step
            if args.method in ('optiq','direct_gmm'):
                import jax
                import jax.numpy as jnp
                import flax.serialization
                from optiq_dime.policy import OptiQPolicy
                key=jax.random.PRNGKey(92001)
                sample=jax.jit(lambda state,obs,key:OptiQPolicy.sample_action(state,obs,key,False,True))
                def act(obs):
                    nonlocal key
                    key,k=jax.random.split(key)
                    return np.asarray(sample(self.model.policy.actor_state,jnp.asarray(obs),k))
                @jax.jit
                def value(state,obs,a):
                    # The v5 teacher uses live mean-Q; TD continues to use target min-Q.
                    return state.apply_fn({'params':state.params,'batch_stats':state.batch_stats},obs,a,train=False).mean(0).squeeze(-1)
                def q(obs,a):return np.asarray(value(self.model.policy.qf_state,obs,a))
                state=dict(actor=self.model.policy.actor_state,target_actor=self.model.policy.target_actor_state,critic=self.model.policy.qf_state,
                           entropy=self.model.ent_coef_state,key=jax.random.key_data(self.model.key),policy_key=jax.random.key_data(self.model.policy.key),noise_key=jax.random.key_data(self.model.policy.noise_key),updates=step)
                (folder/'checkpoints'/f'update_{step:07d}.bin').write_bytes(flax.serialization.to_bytes(state))
                temperature=args.temperature
            else:
                import torch
                def act(obs):return self.model.predict(obs,deterministic=False)[0]
                def q(obs,a):
                    with torch.no_grad():
                        values=self.model.critic(torch.as_tensor(obs,device='cuda'),torch.as_tensor(a,device='cuda'))
                        return torch.minimum(*values).cpu().numpy().reshape(-1)
                self.model.save(folder/'checkpoints'/f'update_{step:07d}.zip')
                temperature=float(self.model.log_ent_coef.detach().exp().cpu())
            self.model.save_replay_buffer(folder/'checkpoints'/f'update_{step:07d}.replay.pkl')
            info=dict(elapsed_seconds=time.monotonic()-self.started)
            log_path=folder/'training_logs/progress.csv'
            if log_path.exists():
                for row in csv.DictReader(log_path.open()):
                    for metric_key,val in row.items():
                        if val and metric_key.startswith('train/'):
                            try:info[metric_key]=float(val)
                            except ValueError:pass
            with evaluation_rng():
                evaluate_navigation(folder,args.name,step,self.model.num_timesteps,act,q,args.eval_episodes,
                                    info,temperature=temperature)
            atomic_json(folder/'status.json',dict(status='training',pid=os.getpid(),updates=step,env_steps=self.model.num_timesteps))
        def _on_training_start(self):self.snapshot()
        def _on_step(self):
            step=self.model._n_updates
            if step and step%10000==0 and step!=self.last:self.snapshot()
            if self.model.num_timesteps%1000==0:
                atomic_json(folder/'status.json',dict(status='training',pid=os.getpid(),updates=step,env_steps=self.model.num_timesteps))
                print(json.dumps(dict(event='navigation_train',method=args.method,updates=step,env_steps=self.model.num_timesteps)),flush=True)
            return True
        def _on_training_end(self):self.snapshot()
    model.learn(total_timesteps=args.steps+args.warmup,callback=Callback(),progress_bar=False)
    assert model._n_updates==args.steps,(model._n_updates,args.steps)
    atomic_json(folder/'status.json',dict(status='completed',pid=os.getpid(),updates=model._n_updates,step=model._n_updates,env_steps=model.num_timesteps))
    env.close()


def run_custom(args,folder):
    from .navigation_agents import DIPOOnline,MEowOnline,MFPOOnline
    env=GMM40Navigation();env.action_space.seed(args.seed)
    agent={'dipo':DIPOOnline,'meow':MEowOnline,'mfpo':MFPOOnline}[args.method](env,args.seed,folder)
    from .model_sizes import save_sizes
    if args.method=='dipo':save_sizes(folder,actor=agent.agent.actor,critic=agent.agent.critic)
    elif args.method=='meow':save_sizes(folder,joint_flow_QV=agent.agent.policy)
    else:save_sizes(folder,actor=agent.agent.actor,divergence=agent.agent.logp_mvel,critic_1=agent.agent.critic_1,critic_2=agent.agent.critic_2)
    if args.method=='meow':
        agent.agent.args.alpha=args.temperature
        agent.agent.policy.alpha=args.temperature
        agent.agent.policy_old.alpha=args.temperature
    obs,_=env.reset(seed=args.seed)
    started=time.monotonic();last=-1
    def snapshot(env_steps,info):
        nonlocal last
        if agent.updates==last:return
        last=agent.updates
        agent.save(folder/'checkpoints'/f'update_{last:07d}.bin')
        with (folder/'checkpoints'/f'update_{last:07d}.environment.pkl').open('wb') as f:
            pickle.dump(dict(observation=obs,env_state=env.state,elapsed=env.elapsed,env_rng=env.np_random.bit_generator.state,action_rng=env.action_space.np_random.bit_generator.state),f)
        if args.method=='mfpo':
            import jax
            agent.eval_key=jax.random.PRNGKey(92001)
            temperature=float(agent.agent.temp.apply_fn({'params':agent.agent.temp.params}))
        else:temperature=args.temperature if args.method=='meow' else None
        try:
            with evaluation_rng():evaluate_navigation(folder,args.name,last,env_steps,agent.act,agent.q,args.eval_episodes,info,temperature=temperature)
        finally:
            if args.method=='mfpo':agent.eval_key=None
    snapshot(0,{})
    for t in range(args.steps+args.warmup):
        action=env.action_space.sample() if t<args.warmup else agent.act(obs[None])[0]
        next_obs,reward,done,truncated,_=env.step(action)
        agent.store(obs,action,reward,next_obs,done)
        obs=next_obs
        if done:obs,_=env.reset()
        info={}
        if t>=args.warmup:
            info=agent.update()
            if not all(np.isfinite(v) for v in info.values()):raise FloatingPointError(info)
            if agent.updates%10000==0 or agent.updates==args.steps:snapshot(t+1,info)
        if (t+1)%1000==0:
            state=dict(status='training',pid=os.getpid(),step=agent.updates,updates=agent.updates,env_steps=t+1,elapsed_seconds=time.monotonic()-started,metrics=info)
            atomic_json(folder/'status.json',state);print(json.dumps(dict(event='navigation_train',method=args.method,**state)),flush=True)
            with (folder/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(state,allow_nan=False)+'\n')
    assert agent.updates==args.steps
    atomic_json(folder/'status.json',dict(status='completed',pid=os.getpid(),step=agent.updates,updates=agent.updates,env_steps=args.steps+args.warmup))
    env.close()
