"""Controlled inference from identical physical states; no training or updates."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',
                  OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',USE_FLAX='0',USE_TORCH='1')
cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[:4])
import argparse,copy,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
import jax
import jax.numpy as jnp
import flax.serialization as fs
from omegaconf import OmegaConf
import gymnasium as gym

P=argparse.ArgumentParser();P.add_argument('--output',type=Path,required=True);a=P.parse_args()
OUT=a.output;OUT.mkdir(parents=True,exist_ok=False)
SOURCE=Path('/home/heechan/OptiQ-ops/sources/484f92e7d6d34c964d85b4493ff17c5a9ebcf32e')
RUN=Path('/home/heechan/optiq-experiments/antmaze-optiq-dense-dacer-off-T1-s0-20260924/runs/v3-optiq-dacer-off-T1-s0')
sys.path.insert(0,str(SOURCE))
from antmaze_experiments.envs import vector,transition,make_one
cfg=json.loads((RUN/'config.json').read_text())
assert cfg['temperature']==1 and not cfg['noveld_enabled'] and not cfg['dacer_enabled']
start_time=time.monotonic()
def emit(event,**kw):
    row=dict(event=event,seconds=time.monotonic()-start_time,**kw)
    print(json.dumps(row,allow_nan=False),flush=True)
    (OUT/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
class Descriptor(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(29,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,shape=(8,),dtype=np.float32)
    def reset(self,**kwargs):raise RuntimeError('Descriptor cannot collect')
    def step(self,a):raise RuntimeError('Descriptor cannot collect')
spec=importlib.util.spec_from_file_location('frozen_train',SOURCE/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
native=OmegaConf.create(cfg['native']);native.output_root=str(OUT/'constructor')
model=module.runner.OptiQDIME('MlpPolicy',env=Descriptor(),cfg=native,model_save_path=None,save_every_n_steps=cfg['steps'])
p=model.policy
template=dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state)
states={};proofs={};state_hashes={}
for label,step in [('2M',2000128),('2.75M',2750208),('final',4008448)]:
    path=RUN/'checkpoint-final.pt' if label=='final' else RUN/'policy-checkpoints'/f'step_{step:010d}'/'policy.pt'
    proof=json.loads((RUN/'checkpoint-verification.json' if label=='final' else path.with_name('verification.json')).read_text())
    sha=digest(path);assert sha==proof['sha256']
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['step']==step and payload['config']['source_commit']==cfg['source_commit']
    states[label]=fs.from_bytes(template,payload['learner']['policy'])
    raw=fs.msgpack_restore(payload['learner']['policy'])
    def equal(x,y):
        if isinstance(y,dict):
            assert set(x)==set(y)
            for k in y:equal(x[k],y[k])
        else:np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    equal(fs.to_state_dict(states[label]),raw)
    state_hashes[label]=hashlib.sha256(fs.to_bytes(states[label])).hexdigest()
    proofs[label]=dict(path=str(path),sha256=sha,step=step,parameters_restored_exact=True)
    del payload,raw;gc.collect()
    emit('checkpoint_loaded',label=label,step=step)
assert all(d.platform=='cpu' for d in jax.devices())

def actions(label,obs):
    p.reset_noise()
    return np.asarray(p.sample_action(states[label]['actor'],obs,p.noise_key,deterministic=False,sample_conditional_noise=True))
def summary(xy,obs,returns,lengths,goals,starts,labels=None):
    rows=[]
    for i,n in enumerate(lengths):
        line=xy[i,:n+1];o=obs[i,:n+1]
        distances=np.linalg.norm(line[:,None,:]-np.array([[-12,12],[12,-12]])[None,:,:],axis=-1)
        rewards=-distances[1:].min(1)
        np.testing.assert_allclose(rewards.sum(),returns[i],rtol=2e-6,atol=.01)
        q=o[:,3:7];up=1-2*(q[:,1]**2+q[:,2]**2)
        left=bool((line[:,0]<-8).any());right=bool((line[:,0]>8).any())
        rows.append(dict(episode=i,source=None if labels is None else labels[i],route='both' if left and right else 'left' if left else 'right' if right else 'central',
            success=bool(goals[i]),goal=int(goals[i]),length=int(n),return_undiscounted=float(returns[i]),return_discounted=float(np.sum(rewards*.99**np.arange(n))),
            closest_goal=float(distances.min()),closest_left_goal=float(distances[:,0].min()),endpoint=line[-1].tolist(),
            final_height=float(o[-1,2]),final_upright_cos=float(up[-1]),last100_height_median=float(np.median(o[-100:,2])),
            last100_upright_median=float(np.median(up[-100:])),fraction_height_below_02=float((o[:,2]<.2).mean()),
            start=starts[i].tolist()))
    return rows

# Re-run the old checkpoint with the original 40 initial-state draw/RNG schedule.
# CPU arithmetic can diverge from the saved GPU rollout, so this is fresh inference,
# not a claim of bitwise trajectory reproduction. Preserve all 40 outcomes.
env=vector('v3',20,seed=87231,asynchronous=False,reward_profile='dense',random_init=True)
p.key=jax.random.PRNGKey(700000+2000128)
xy=np.full((40,701,2),np.nan,np.float32);ob=np.full((40,701,29),np.nan,np.float32)
rets=np.zeros(40);lengths=np.zeros(40,dtype=int);goals=np.zeros(40,dtype=int)
starts=[];snapshots={}
for batch in range(2):
    observations=env.reset();active=np.ones(20,bool)
    for i,e in enumerate(env.envs):
        ix=batch*20+i;st=e.state();starts.append(np.r_[st['qpos'],st['qvel']]);snapshots[ix]={'start':copy.deepcopy(st)}
        ob[ix,0]=observations[i];xy[ix,0]=observations[i,:2]
    for tick in range(700):
        act=actions('2M',observations);nxt,reward,done,info=env.step(act);final,_=transition(nxt,done,info)
        for i in np.flatnonzero(active):
            ix=batch*20+i;lengths[ix]+=1;n=lengths[ix]
            ob[ix,n]=final[i];xy[ix,n]=final[i,:2];rets[ix]+=reward[i]
            if not done[i]:
                if final[i,0]<-8 and 'left_gate' not in snapshots[ix]:snapshots[ix]['left_gate']=copy.deepcopy(env.envs[i].state())
                if n==600:snapshots[ix]['late600']=copy.deepcopy(env.envs[i].state())
            else:active[i]=False;goals[ix]=int(info[i].get('success',0))
        observations=nxt
        if not active.any():break
    emit('old_policy_replay_batch',episodes=(batch+1)*20)
env.close();starts=np.array(starts)
old_rows=summary(xy,ob,rets,lengths,goals,starts)
saved=np.load(RUN/'evaluations/0002000128/policy-natural/rollouts.npz')
np.testing.assert_allclose(starts[:20],saved['initial_full_state'][:20],atol=1e-10)
np.savez_compressed(OUT/'old_policy_replay.npz',xy=xy,observations=ob,returns=rets,lengths=lengths,goals=goals,initial_full_state=starts)
write('old_policy_replay.json',old_rows)
left=[r['episode'] for r in old_rows if r['route']=='left' and set(('start','left_gate','late600')).issubset(snapshots[r['episode']])]
assert len(left)>=4,dict(left_candidates=left)
selected=[left[i] for i in np.linspace(0,len(left)-1,4,dtype=int)]
bank=[]
for episode in selected:
    for stage in ('start','left_gate','late600'):
        state=snapshots[episode][stage]
        bank.append(dict(id=f'old_episode{episode}-{stage}',episode=episode,stage=stage,state=state))
center=make_one('v3',91030,reward_profile='dense',random_init=False);center.reset()
bank.append(dict(id='training-center',episode=-1,stage='training-center',state=copy.deepcopy(center.state())))
center.close()
torch.save(bank,OUT/'physical_state_bank.pt')
bank_info=[]
for b in bank:
    st=b['state'];q=st['qpos'][3:7]
    bank_info.append(dict(id=b['id'],stage=b['stage'],episode=b['episode'],xy=st['qpos'][:2].tolist(),
        height=float(st['qpos'][2]),upright_cos=float(1-2*(q[1]**2+q[2]**2)),elapsed=int(st['elapsed']),remaining=700-int(st['elapsed'])))
write('state_bank.json',bank_info);emit('state_bank_ready',left_candidates=len(left),selected=selected,states=len(bank))

# Three stochastic repetitions per physical state, paired across checkpoints.
all_results={};replicates=3;labels=[b['id'] for b in bank for _ in range(replicates)]
envs=[make_one('v3',820000+i,reward_profile='dense',random_init=True) for i in range(len(labels))]
for label in states:
    obs=[];initial=[]
    for e,b in zip(envs,[b for b in bank for _ in range(replicates)]):
        e.reset();o=e.restore(copy.deepcopy(b['state']));obs.append(o);initial.append(np.r_[b['state']['qpos'],b['state']['qvel']])
    obs=np.array(obs);initial=np.array(initial);count=len(envs)
    xx=np.full((count,701,2),np.nan,np.float32);oo=np.full((count,701,29),np.nan,np.float32)
    xx[:,0]=obs[:,:2];oo[:,0]=obs
    rr=np.zeros(count);ll=np.zeros(count,dtype=int);gg=np.zeros(count,dtype=int);active=np.ones(count,bool)
    p.key=jax.random.PRNGKey(6220941)
    for tick in range(700):
        act=actions(label,obs)
        for i in np.flatnonzero(active):
            obs[i],reward,done,info=envs[i].step(act[i]);ll[i]+=1
            xx[i,ll[i]]=obs[i,:2];oo[i,ll[i]]=obs[i];rr[i]+=reward
            if done:active[i]=False;gg[i]=int(info.get('success',0))
        if tick%200==199:emit('continuation_progress',label=label,ticks=tick+1,active=int(active.sum()))
        if not active.any():break
    assert not active.any()
    result=summary(xx,oo,rr,ll,gg,initial,labels)
    all_results[label]=result
    np.savez_compressed(OUT/f'continuation_{label}.npz',xy=xx,observations=oo,returns=rr,lengths=ll,goals=gg,initial_full_state=initial,source_labels=labels)
    write(f'continuation_{label}.json',result)
    emit('continuation_completed',label=label,episodes=count,successes=int((gg>0).sum()))
for e in envs:e.close()

# Actor and critic crossed on exactly the same physical states/action candidates.
obs=np.array([np.r_[b['state']['qpos'],b['state']['qvel']] for b in bank],np.float32)
N=256;repeated=np.repeat(obs,N,axis=0);candidates={};qmatrix={}
for label in states:
    candidates[label]=np.asarray(p.sample_action(states[label]['actor'],repeated,jax.random.PRNGKey(240925),deterministic=False,sample_conditional_noise=True))
@jax.jit
def qvalues(params,batch_stats,obs,act):
    return p.qf_state.apply_fn({'params':params,'batch_stats':batch_stats},obs,act,train=False,rngs={'dropout':jax.random.PRNGKey(11)})
for critic_label,state in states.items():
    qmatrix[critic_label]={}
    for actor_label,act in candidates.items():
        q=np.asarray(qvalues(state['critic'].params,state['critic'].batch_stats,repeated,act)).reshape(2,len(bank),N)
        assert np.isfinite(q).all()
        qmatrix[critic_label][actor_label]=dict(mean_q=q.mean((0,2)).tolist(),min_q=q.min(0).mean(1).tolist(),twin_disagreement=np.abs(q[0]-q[1]).mean(1).tolist())
write('q_cross_comparison.json',dict(states=bank_info,samples_per_actor_state=N,comparison=qmatrix))
for label in states:assert hashlib.sha256(fs.to_bytes(states[label])).hexdigest()==state_hashes[label]
for label,pr in proofs.items():assert digest(Path(pr['path']))==pr['sha256']
write('verification.json',dict(passed=True,checkpoints_unchanged=True,model_and_optimizer_unchanged=True,
    paired_full_physical_states=True,preserved_episode_clock=True,paired_random_seeds=True,
    source=cfg['source_commit'],checkpoint_proofs=proofs,noveld=False,dacer_noise=False,
    modes='fresh random z + conditional sigma',runtime_devices=[str(d) for d in jax.devices()],
    artifacts={f.name:digest(f) for f in OUT.glob('*.npz')}))
write('result.json',dict(completed=True,seconds=time.monotonic()-start_time,selected_old_left_episodes=selected,
    repetitions_per_state=replicates,states=bank_info,continuations=all_results,
    limitations=['Four evenly selected left-moving replay episodes; a diagnostic, not a seed-averaged outcome.',
    'Fresh CPU inference may numerically diverge from original GPU trajectories.',
    'Finite horizon returns omit beyond-time-limit bootstrap; critic values are not directly the same estimand.',
    'Changing the entire checkpoint changes actor and critic together during evaluation; Q crossing is diagnostic only.']))
emit('complete',output=str(OUT))
