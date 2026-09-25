"""Uniform dataset -> IQL critic -> frozen-Q existing Direct-GMM extraction.

IQL equations/order: ikostrikov/implicit_q_learning critic.py, learner.py.
This adapts critic learning to our existing scalar GELU twin architecture;
it is NOT the official IQL actor (AWR), and not an official IQL reproduction.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2))


def collect(folder, count):
    from .envs import make_one
    env=make_one('v1',0,reward_profile='progress100_euclidean_no_bonus')
    rng=np.random.default_rng(0)
    obs=env.reset()
    arrays={k:np.empty((count,dim),np.float32) for k,dim in
            [('observations',29),('actions',8),('next_observations',29)]}
    arrays.update(rewards=np.empty(count,np.float32),masks=np.empty(count,np.float32),
                  episode_ends=np.zeros(count,bool),success=np.zeros(count,bool))
    episodes=0
    try:
        for i in range(count):
            action=rng.uniform(-1,1,8).astype(np.float32)
            nxt,reward,done,info=env.step(action)
            arrays['observations'][i]=obs;arrays['actions'][i]=action
            arrays['next_observations'][i]=nxt;arrays['rewards'][i]=reward
            arrays['masks'][i]=float(not done or info.get('TimeLimit.truncated',False))
            arrays['episode_ends'][i]=done;arrays['success'][i]=bool(info.get('success',0))
            episodes+=int(done)
            obs=env.reset() if done else nxt
            if (i+1)%10000==0:
                write(folder/'progress.json',dict(stage='uniform_collection',transitions=i+1,
                    successes=int(arrays['success'][:i+1].sum()),episodes=episodes))
    finally: env.close()
    assert np.isfinite(arrays['observations']).all()
    assert np.max(np.abs(arrays['actions']))<=1
    assert np.all(arrays['masks'][arrays['success']]==0)
    assert np.all(arrays['masks'][arrays['episode_ends']&~arrays['success']]==1)
    np.savez_compressed(folder/'uniform_dataset.npz',**arrays)
    xy=arrays['next_observations'][:,:2]
    stats=dict(transitions=count,episodes=episodes,successes=int(arrays['success'].sum()),
        action_mean=arrays['actions'].mean(0).tolist(),action_std=arrays['actions'].std(0).tolist(),
        xy_min=xy.min(0).tolist(),xy_max=xy.max(0).tolist(),
        left_of_obstacle=int((xy[:,0]<-6).sum()),reward='100*delta_distance-1',entropy_bonus=0)
    write(folder/'dataset_summary.json',stats)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .progress_reward import maze_geometry
    from matplotlib.patches import Rectangle
    walls,goals,bounds=maze_geometry('v1')
    fig,ax=plt.subplots(figsize=(7,7))
    ax.hist2d(*xy.T,bins=100,range=[bounds[[0,2]],bounds[[1,3]]],norm=LogNorm())
    for x,y,u,v in walls: ax.add_patch(Rectangle((x,y),u-x,v-y,color='.7'))
    ax.scatter(*goals.T,marker='*',color='red');ax.set_aspect('equal')
    ax.set(title=f'Uniform random data: {count:,} transitions',xlabel='x (m)',ylabel='y (m)')
    fig.savefig(folder/'dataset_coverage.png',dpi=160);plt.close(fig)
    return arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    folder=a.output;folder.mkdir(parents=True,exist_ok=False)
    count,qsteps,asteps=(2048,8,8) if a.smoke else (1000000,250000,100000)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    config=dict(source_commit=source,seed=0,transitions=count,iql_updates=qsteps,
        extraction_updates=asteps,expectile=.9,gamma=.99,tau=.005,learning_rate=.0003,
        temperature=1.,batch_size=256,entropy_bonus=0,offline=True,
        critic='existing scalar GELU 256x2 twin',value='ReLU 256x2',
        actor='existing DirectGMM/TRG N=M64; NOT IQL AWR',
        reward='100*(d_current-d_next)-1',target_q_aggregation='min',
        extraction_q_aggregation='existing config',evaluation='fixed origin, random z, mu-only')
    write(folder/'protocol.json',config)
    data=collect(folder,count)
    # Collection is CPU-only. Wait for an existing campaign worker to release a GPU.
    import fcntl
    locks=[]
    while True:
        for gpu in range(4):
            handle=open(f'/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock','a')
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: handle.close();continue
            locks.append(handle);break
        if locks: break
        write(folder/'progress.json',dict(stage='dataset_complete_waiting_for_gpu',transitions=count))
        time.sleep(10)
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import flax.serialization as fs
    from flax.training.train_state import TrainState
    import optax
    from .envs import make_one
    from .learners import JaxLearner
    from .run import evaluate
    from .xy_entropy import plot_trace
    env=make_one('v1',0);spaces=(env.observation_space,env.action_space);env.close()
    learner=JaxLearner('optiq',spaces,'v1',folder,temperature=1.,budget=1000000,
        reward_profile='progress100_euclidean_no_bonus',noveld=False,dacer_enabled=False,
        collection_profile='single-update1',optiq_config_profile='basic')
    learner.eval_fixed_starts=True;learner.eval_random_starts=False
    model=learner.model;policy=model.policy
    class Value(nn.Module):
        @nn.compact
        def __call__(self,s):
            s=nn.relu(nn.Dense(256)(s));s=nn.relu(nn.Dense(256)(s))
            return nn.Dense(1)(s)[...,0]
    net=Value();value=TrainState.create(apply_fn=net.apply,
        params=net.init(jax.random.PRNGKey(42),jnp.zeros((1,29)))['params'],tx=optax.adam(3e-4))
    def qapply(state,params,s,act):
        q=state.apply_fn({'params':params,'batch_stats':state.batch_stats},s,act,train=False)
        return q[...,0]
    @jax.jit
    def iql(q,v,s,act,nxt,r,mask):
        target_q=jax.lax.stop_gradient(qapply(q,q.target_params,s,act).min(axis=0))
        def vl(params):
            diff=target_q-v.apply_fn({'params':params},s)
            return jnp.mean(jnp.where(diff>0,.9,.1)*diff**2)
        vloss,vg=jax.value_and_grad(vl)(v.params);v=v.apply_gradients(grads=vg)
        target=jax.lax.stop_gradient(r+.99*mask*v.apply_fn({'params':v.params},nxt))
        def ql(params): return jnp.mean(jnp.sum((qapply(q,params,s,act)-target[None])**2,axis=0))
        qloss,qg=jax.value_and_grad(ql)(q.params);q=q.apply_gradients(grads=qg)
        q=q.replace(target_params=jax.tree.map(lambda x,y:.005*x+.995*y,q.params,q.target_params))
        return q,v,qloss,vloss
    rng=np.random.default_rng(123)
    def batch():
        ix=rng.integers(count,size=256)
        return [jnp.asarray(data[k][ix]) for k in ('observations','actions','next_observations','rewards','masks')]
    actor_cfg=model.cfg.alg.actor
    signature=inspect.signature(model.update_actor)
    kwargs={k:actor_cfg[k] for k in signature.parameters if k in actor_cfg}
    kwargs.update(z_atoms=jnp.linspace(model.cfg.alg.critic.v_min,model.cfg.alg.critic.v_max,
        model.cfg.alg.critic.n_atoms),semi_implicit=True)
    # Real-data scratch preflight; never changes the actual initial model.
    check_batch=batch()
    cq,cv,cql,cvl=iql(policy.qf_state,value,*check_batch)
    assert np.isfinite([float(cql),float(cvl)]).all()
    ca,_,_,cm=model.update_actor(actor_state=policy.actor_state,qf_state=cq,
        observations=check_batch[0],key=jax.random.PRNGKey(987),**kwargs)
    assert int(ca.step)==int(policy.actor_state.step)+1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ca.params))
    write(folder/'preflight.json',dict(passed=True,q_loss=float(cql),value_loss=float(cvl),
        scratch_actor_update=True,gpu=gpu,initial_actor_unmodified=True))
    import wandb
    run=wandb.init(entity='OptiQ',project='jaehun-antmaze',group='uniform1M-IQL-frozenQ',
        name='v1-uniform1M-IQL250k-extract100k-T1-s0',config=config,dir=str(folder),
        mode='disabled' if a.smoke else 'offline')
    write(folder/'wandb.json',dict(id=run.id,project='jaehun-antmaze',mode='offline'))
    run.log({'dataset/coverage':wandb.Image(str(folder/'dataset_coverage.png'))},step=0)
    actor_initial=hashlib.sha256(fs.to_bytes(policy.actor_state)).hexdigest()
    q_initial=hashlib.sha256(fs.to_bytes(policy.qf_state)).hexdigest()
    try:
        for step in range(1,qsteps+1):
            policy.qf_state,value,ql,vl=iql(policy.qf_state,value,*batch())
            if step%1000==0 or step==qsteps:
                metrics=dict(stage='iql',updates=step,q_loss=float(ql),value_loss=float(vl))
                assert np.isfinite([float(ql),float(vl)]).all()
                write(folder/'progress.json',metrics);run.log({'iql/'+k:v for k,v in metrics.items()},step=step)
        assert hashlib.sha256(fs.to_bytes(policy.actor_state)).hexdigest()==actor_initial
        assert hashlib.sha256(fs.to_bytes(policy.qf_state)).hexdigest()!=q_initial
        frozen=fs.to_bytes(policy.qf_state);(folder/'iql_critic.msgpack').write_bytes(frozen)
        (folder/'iql_value.msgpack').write_bytes(fs.to_bytes(value))
        key=jax.random.PRNGKey(100)
        for step in range(1,asteps+1):
            obs=batch()[0]
            policy.actor_state,_,key,metrics=model.update_actor(actor_state=policy.actor_state,
                qf_state=policy.qf_state,observations=obs,key=key,**kwargs)
            if step%1000==0 or step==asteps:
                scalar={k:float(v) for k,v in metrics.items() if np.asarray(v).size==1}
                assert all(np.isfinite(v) for v in scalar.values())
                run.log({'extraction/'+k:v for k,v in scalar.items()},step=qsteps+step)
                write(folder/'progress.json',dict(stage='extraction',updates=step))
            if step%5000==0 or step==asteps:
                assert fs.to_bytes(policy.qf_state)==frozen
                summary=evaluate(learner,'v1',folder,step,2 if a.smoke else 100,'native',True)
                run.log({'eval/success_rate':summary['success_rate'],
                    'eval/return':summary['mean_return'],
                    'eval/center_100_trajectories':wandb.Image(str(plot_trace(folder,step)))},step=qsteps+step)
                (folder/f'actor_{step:06d}.msgpack').write_bytes(fs.to_bytes(policy.actor_state))
        assert fs.to_bytes(policy.qf_state)==frozen
        assert hashlib.sha256(fs.to_bytes(policy.actor_state)).hexdigest()!=actor_initial
        write(folder/'result.json',dict(completed=True,critic_frozen_verified=True,
            iql_updates=qsteps,extraction_updates=asteps,transitions=count,final_evaluation=summary))
    finally: run.finish()

if __name__=='__main__':main()
