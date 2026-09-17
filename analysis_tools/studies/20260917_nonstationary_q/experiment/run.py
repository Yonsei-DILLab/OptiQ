"""Finite registered runs with complete restart state and immutable prefix forks."""
import argparse,json,time,signal,traceback,os,shutil
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from .config import WARMUP,GRID_N,eval_step,raw_step,write_json
from .models import actor_state,critic_state,td_update,q_mean,q_reference_mean,sample_action
from .problems import GRID,schedule,PositionChoice
from .actor import engine
from .diagnostics import evaluate,interpolation_check,policy_returns
from .io import save,save_array,checkpoint,sha,verify_source,complete

STOP=False
def stop(*args):
    global STOP
    STOP=True

class Replay:
    def __init__(self,seed):
        self.data={k:np.zeros((1000000,1),np.float32) for k in ('s','a','sp','r')}
        self.size=0;self.rng=np.random.default_rng(50100+seed)
    def add(self,s,a,sp,r):
        for k,v in zip(self.data,(s,a,sp,r)):self.data[k][self.size]=v
        self.size+=1
    def sample(self):
        ix=self.rng.integers(self.size,size=256)
        return {k:jnp.asarray(v[ix,0] if k=='r' else v[ix]) for k,v in self.data.items()}
    def state(self):return dict(data={k:v[:self.size] for k,v in self.data.items()},size=self.size,rng=json.dumps(self.rng.bit_generator.state))
    def restore(self,d):
        self.size=int(d['size'])
        for k,v in d['data'].items():self.data[k][:self.size]=v
        self.rng.bit_generator.state=json.loads(d['rng'])

class QStream:
    def __init__(self,root):self.root=Path(root);self.index=None;self.array=None
    def at(self,t):
        if t==0:return jnp.asarray(np.load(self.root/'initial.npy'))
        first=1+200*((t-1)//200)
        if self.index!=first:
            self.array=np.load(self.root/f'{first:06d}.npy',mmap_mode='r');self.index=first
        return jnp.asarray(np.asarray(self.array[t-first]))

def run(root,task,validation=False,max_updates=None,warmup=WARMUP,interrupt_at=None):
    global STOP
    STOP=False
    root=Path(root);code=verify_source(root)
    base=root/('validation_runs' if validation else 'runs');out=base/task['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        assert json.loads((out/'COMPLETE.json').read_text())['source_code_id']==code
        return
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    write_json(out/'RUNNING.json',dict(pid=os.getpid(),host=os.uname().nodename,time=time.time(),task=task))
    stage=task['stage'];rl=stage in ('source','closed');qkind='critic' if rl else ('replay' if stage=='replay' else 'analytic')
    nsteps=max_updates or task['updates'];f=engine(task['method'],task['n'],task['m'],qkind)
    state=actor_state(task['method'],task['seed']);critic=critic_state(task['seed']) if rl else None
    key=jax.random.PRNGKey(1000+task['seed']);collectkey=jax.random.PRNGKey(201000+task['seed'])
    env=PositionChoice();replay=Replay(task['seed']) if rl else None
    warmrng=np.random.default_rng(88001+task['seed']);start=0;elapsed=0.;parent_hash=None
    stream=None
    if stage=='replay':
        src=base/task['q_source'];record=json.loads((src/'COMPLETE.json').read_text())
        assert record['source_code_id']==code and record['step']>=nsteps
        stream=QStream(src/'qstream')
        write_json(out/'Q_SOURCE.json',dict(name=task['q_source'],complete_sha256=sha(src/'COMPLETE.json'),code_id=code,
            representation='live twin-mean Q(0,a), highest precision reference matmul, 8193 grid values every critic update; linear action interpolation, no time interpolation'))
    restore=out/'checkpoint.msgpack'
    if not restore.exists() and task.get('parent'):
        parent=base/task['parent'];record=json.loads((parent/'COMPLETE.json').read_text())
        assert record['source_code_id']==code and (validation or record['step']==20000)
        restore=parent/'checkpoint.msgpack';parent_hash=sha(restore)
        write_json(out/'PARENT.json',dict(name=task['parent'],checkpoint_sha256=parent_hash,
            shared_prefix='First 20000 actor updates and all prefix evaluations are in the parent; actor, optimizer and RNG are copied exactly.'))
    if restore.exists():
        ck=serialization.msgpack_restore(restore.read_bytes());assert ck['source_code_id']==code
        state=serialization.from_state_dict(state,ck['actor']);key=jnp.asarray(ck['key']);collectkey=jnp.asarray(ck['collectkey'])
        start=int(ck['step']);elapsed=float(ck['elapsed'])
        if rl:
            critic=serialization.from_state_dict(critic,ck['critic']);replay.restore(ck['replay'])
            env.state=float(ck['env_state']);env.episode_step=int(ck['episode_step'])
    initial=serialization.to_bytes(state.params)
    if not (out/'initial_actor.msgpack').exists():(out/'initial_actor.msgpack').write_bytes(initial)
    write_json(out/'config.json',dict(**task,actual_updates=nsteps,warmup=warmup if rl else 0,
        source_code_id=code,temperature=.25,initial_state_step=start if parent_hash else 0,
        actor='v5 256x2 GELU conditional Gaussian',critic='v5 256x2 GELU scalar twin' if rl else 'none',
        critic_batch=256 if rl else None,
        proposal='conditional Gaussian mixture, IID components, teacher-only sigma floor .05; no anchors',
        gamma=.99 if rl else None,learning_rate=.0003,grid_points=GRID_N,
        evaluation='32768 fresh latent / conditional noise pairs; histogram 512 bins; no smoothing',
        validation=validation))
    if rl and start==0 and replay.size==0:
        for _ in range(warmup):
            s=env.state;a=float(warmrng.uniform(-1,1));sp,r,terminal,timeout=env.step(a);replay.add(s,a,sp,r)
        if stage=='source':
            q=np.asarray(q_reference_mean(critic.params,jnp.zeros((1,1)),jnp.asarray(GRID,jnp.float32)[None,:,None]))[0]
            save_array(out/'qstream/initial.npy',q)
    qarg=(critic.params if rl else stream.at(start) if stream else jnp.asarray(schedule(stage,start)))
    if start==0:evaluate(out,task,0,state,f,qarg,qkind)
    rows=[];evaluations=[];qrows=[];chunk=start+1;begin=time.time();last=start
    def flush(step):
        nonlocal rows,evaluations,qrows,chunk
        if rows:save(out/'metrics'/f'{chunk:06d}.npz',{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        if evaluations:save(out/'evaluations'/f'{chunk:06d}.npz',{k:np.stack([r[k] for r in evaluations]) for k in evaluations[0]})
        if qrows:
            # Fixed block names allow partial block overwrite after a checkpoint restart.
            block=1+200*((chunk-1)//200);path=out/'qstream'/f'{block:06d}.npy'
            prefix=np.load(path)[:chunk-block] if chunk>block else np.empty((0,GRID_N),np.float32)
            save_array(path,np.concatenate([prefix,np.stack(qrows)]))
        ck=dict(source_code_id=code,actor=serialization.to_state_dict(state),key=np.asarray(key),collectkey=np.asarray(collectkey),
            step=step,elapsed=elapsed+time.time()-begin)
        if rl:ck.update(critic=serialization.to_state_dict(critic),replay=replay.state(),env_state=env.state,episode_step=env.episode_step)
        checkpoint(out/'checkpoint.msgpack',ck)
        stats=rows[-1] if rows else {}
        write_json(out/'progress.json',dict(name=task['name'],step=step,total=nsteps,env_steps=warmup+step if rl else None,
            seconds=ck['elapsed'],gpu=str(jax.devices()[0]),source_code_id=code,
            loss=float(stats.get('loss',0)),ess=float(stats.get('ess',0))))
        rows=[];evaluations=[];qrows=[];chunk=step+1
    for step in range(start+1,nsteps+1):
        tick=time.perf_counter();cm={};obs=jnp.zeros((1,1),jnp.float32)
        if rl:
            collectkey,ak=jax.random.split(collectkey);s=env.state
            a=float(sample_action(state,jnp.array([[s]],jnp.float32),ak,deterministic=False)[0,0])
            sp,r,terminal,timeout=env.step(a);replay.add(s,a,sp,r);batch=replay.sample()
            critic,metrics,key=td_update(state,critic,batch,key)
            cm={k:float(v) for k,v in jax.device_get(metrics).items()}
            cm.update(collected_reward=r,collected_action=a,collected_state=s)
            obs=batch['s'] if stage=='source' else batch['s'][:1]
            qarg=critic.params
            if stage=='source':qrows.append(np.asarray(q_reference_mean(critic.params,jnp.zeros((1,1)),jnp.asarray(GRID,jnp.float32)[None,:,None]))[0])
        elif stream:qarg=stream.at(step)
        else:qarg=jnp.asarray(schedule(stage,step))
        before=state.params;state,key,t,value,gn=f['step'](state,obs,key,qarg)
        stat={k:np.asarray(v) for k,v in jax.device_get(f['scalars'](state.params,t,value,gn)).items()}
        stat.update(step=np.array(step),train_seconds=np.array(time.perf_counter()-tick),**{k:np.asarray(v) for k,v in cm.items()})
        if not all(np.isfinite(v).all() for v in stat.values()):
            save(out/'NONFINITE.npz',stat);raise FloatingPointError('Non-finite training metric')
        rows.append(stat)
        if eval_step(step) or raw_step(step) or step==nsteps:
            ev=evaluate(out,task,step,state,f,qarg,qkind)
            if rl:
                checks=interpolation_check(critic.params)
                save(out/'grid_checks'/f'{step:06d}.npz',checks)
                ret=policy_returns(state,task['seed']);save(out/'returns'/f'{step:06d}.npz',dict(step=step,**ret))
                ev.update({k:v for k,v in ret.items() if k!='returns'})
            evaluations.append(ev)
        if raw_step(step) or step==nsteps:
            # Freeze the actual pre-update teacher and compare this exact update.
            pre,_=f['common_nll'](before,t);post,_=f['common_nll'](state.params,t)
            heads_after=f['heads'](state.params,t['obs'],t['z'])
            raw={k:np.asarray(v[:1]) for k,v in t.items() if hasattr(v,'shape')}
            raw.update(mu_after=np.asarray(heads_after[0][:1]),log_sigma_after=np.asarray(heads_after[1][:1]),
                marginal_nll_before=float(pre),marginal_nll_after=float(post),step=step,
                actual_actor_batch=obs.shape[0],saved_batch_rows=1)
            save(out/'actual_updates'/f'{step:06d}.npz',raw)
        last=step
        if validation and step==interrupt_at:STOP=True
        if step%200==0 or step==nsteps or STOP:
            flush(step)
            if step%1000==0 or step==nsteps or STOP:print((out/'progress.json').read_text().strip(),flush=True)
        if STOP:raise SystemExit(75)
    if last<nsteps:raise RuntimeError('No training steps executed')
    evaluate(out,task,nsteps,state,f,qarg,qkind,final=True)
    (out/'final_actor.msgpack').write_bytes(serialization.to_bytes(state.params))
    if critic is not None:(out/'final_critic.msgpack').write_bytes(serialization.to_bytes(critic.params))
    complete(out,dict(name=task['name'],stage=stage,step=last,source_code_id=code,finite=True,
        validation=validation,completed_unix=time.time(),actor_step=int(state.step),parent_sha256=parent_hash))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--validation',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--warmup',type=int,default=WARMUP)
    a=p.parse_args();root=Path(a.root);task=json.loads((root/'tasks.json').read_text())[a.index]
    assert jax.default_backend()=='gpu'
    try:run(root,task,a.validation,a.limit,a.warmup)
    except Exception:
        out=root/('validation_runs' if a.validation else 'runs')/task['name']
        write_json(out/'FAILED.json',dict(error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
