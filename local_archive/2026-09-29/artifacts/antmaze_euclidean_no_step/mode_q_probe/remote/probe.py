"""Post-hoc inference only. Never calls train/update or writes to a training run."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def route(task,xy):
    if task=='v3':
        l=(xy[:,0]<-8).any();r=(xy[:,0]>8).any()
        return 'both' if l and r else 'left' if l else 'right' if r else 'uncommitted'
    indices=np.flatnonzero((xy[:-1,0]>-4)&(xy[1:,0]<=-4))
    if not len(indices):return 'uncommitted'
    i=indices[0];f=(-4-xy[i,0])/(xy[i+1,0]-xy[i,0]);y=xy[i,1]+f*(xy[i+1,1]-xy[i,1])
    return 'upper' if y>2 else 'lower' if y<-2 else 'uncommitted'

def stats(x):
    a=np.asarray(x,float);return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['v3','v4'],required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--continuations',type=int,default=4);args=ap.parse_args()
    cpus=sorted(os.sched_getaffinity(0));chosen=cpus[2::4] if args.task=='v3' else cpus[3::4]
    os.sched_setaffinity(0,chosen[:8]);args.output.mkdir(parents=True,exist_ok=False)
    source=Path('/home/heechan/OptiQ-ops/sources/f953d28456d3800860dddb9b9cb91b6bd520ae00')
    campaign=Path('/home/heechan/optiq-experiments/antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2')
    run=campaign/'runs'/f'{args.task}-optiq-euclidean-no-step-B0-T1-s0';cfg=json.loads((run/'config.json').read_text())
    sys.path.insert(0,str(source))
    import torch,jax,jax.numpy as jnp,flax.serialization as fs,gymnasium as gym
    from omegaconf import OmegaConf
    from antmaze_experiments.envs import vector,transition
    torch.set_num_threads(1)
    class Descriptor(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(29,),dtype=np.float32)
        action_space=gym.spaces.Box(-1,1,(8,),dtype=np.float32)
    spec=importlib.util.spec_from_file_location('frozen_trg',source/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=OmegaConf.create(cfg['native']);native.output_root=str(args.output/'constructor')
    model=module.runner.OptiQDIME('MlpPolicy',env=Descriptor(),cfg=native,model_save_path=None,save_every_n_steps=cfg['steps'])
    p=model.policy;template=dict(actor=p.actor_state,critic=p.qf_state,target_actor=p.target_actor_state)
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    assert native.alg.critic.backup_mode=='td' and native.alg.critic.n_atoms==1
    assert native.alg.actor.source_q_eval=='mean'
    hashes={};saved={};steps=[250112,500224,750080]
    for step in steps:
        path=run/'policy-checkpoints'/f'step_{step:010d}'/'policy.pt'
        proof=json.loads((path.parent/'verification.json').read_text());assert digest(path)==proof['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['config']==cfg
        restored=fs.from_bytes(template,payload['learner']['policy']);saved[step]=restored;hashes[str(path)]=digest(path)
    def load(step):
        p.actor_state=saved[step]['actor'];p.qf_state=saved[step]['critic'];p.target_actor_state=saved[step]['target_actor']
    @jax.jit
    def qcall(state,obs,act):
        return state.apply_fn({'params':state.params,'batch_stats':state.batch_stats},obs,act,train=False)[...,0]
    def qall(state,obs,act):
        pieces=[]
        for i in range(0,len(obs),512):
            n=min(512,len(obs)-i);o=np.repeat(obs[i:i+1],512,axis=0);a=np.repeat(act[i:i+1],512,axis=0)
            o[:n]=obs[i:i+n];a[:n]=act[i:i+n];pieces.append(np.asarray(qcall(state,jnp.asarray(o),jnp.asarray(a)))[:,:n])
        return np.concatenate(pieces,axis=1)
    def act(obs):
        p.reset_noise();return np.asarray(p.sample_action(p.actor_state,jnp.asarray(obs),p.noise_key,deterministic=False,sample_conditional_noise=True))
    initial_hash=hashlib.sha256(fs.to_bytes(saved)).hexdigest();started=time.monotonic();load(250112)
    env=vector(args.task,20,seed=87231,asynchronous=False,fixed=True,reward_profile=cfg['reward_profile'],random_init=False)
    episodes=[];p.key=jax.random.PRNGKey(950112)
    for batch in range(2):
        obs=env.reset();common=env.envs[0].initial
        for i,e in enumerate(env.envs):e.initial=common;obs[i]=e.restore(common)
        rows=[dict(obs=[],actions=[],rewards=[],xy=[o[:2].copy()],goal=0,terminated=False) for o in obs];active=np.ones(20,bool)
        for t in range(700):
            action=act(obs);nxt,reward,done,infos=env.step(action);final,terminal=transition(nxt,done,infos)
            for i in np.flatnonzero(active):
                rows[i]['obs'].append(obs[i].copy());rows[i]['actions'].append(action[i].copy());rows[i]['rewards'].append(float(reward[i]));rows[i]['xy'].append(final[i,:2].copy())
                if done[i]:
                    active[i]=False;rows[i]['goal']=int(infos[i].get('success',0));rows[i]['terminated']=bool(terminal[i]);rows[i]['final_obs']=final[i].copy()
            obs=nxt
            if not active.any():break
        episodes.extend(rows);print(json.dumps(dict(phase='reference_rollouts',batch=batch,seconds=time.monotonic()-started)),flush=True)
    env.close()
    for ep in episodes:
        for key in ['obs','actions','rewards','xy']:ep[key]=np.asarray(ep[key])
        ep['route']=route(args.task,ep['xy']);g=0.;rtg=[]
        for r in ep['rewards'][::-1]:g=float(r)+.99*g;rtg.append(g)
        ep['rtg']=np.asarray(rtg[::-1])
    flat_o=np.concatenate([e['obs'] for e in episodes]);flat_a=np.concatenate([e['actions'] for e in episodes]);lengths=np.array([len(e['obs']) for e in episodes]);offsets=np.r_[0,np.cumsum(lengths)]
    bank=dict(observations=flat_o,actions=flat_a,rewards=np.concatenate([e['rewards'] for e in episodes]),return_to_go=np.concatenate([e['rtg'] for e in episodes]),offsets=offsets,routes=np.array([e['route'] for e in episodes]),goals=np.array([e['goal'] for e in episodes]))
    original=np.load(run/'evaluations/0000250112/policy-fixed/rollouts.npz')
    errors=[float(np.max(np.abs(e['xy']-original['xy'][i,:len(e['xy'])]))) if len(e['xy'])==int(original['lengths'][i])+1 else None for i,e in enumerate(episodes)]
    groups={};cross={}
    for step in steps:
        q=qall(saved[step]['critic'],flat_o,flat_a);bank[f'q_{step}']=q
        cross[str(step)]={}
        for side in sorted(set(bank['routes'])):
            ids=[i for i,e in enumerate(episodes) if e['route']==side];data={}
            for t in [0,10,25,50,100,150,200]:
                ix=[offsets[i]+t for i in ids if lengths[i]>t]
                if ix:data[str(t)]=dict(q_mean=stats(q[:,ix].mean(0)),q1=stats(q[0,ix]),q2=stats(q[1,ix]),actual_rtg=stats(bank['return_to_go'][ix]))
            cross[str(step)][side]=data
    for side in sorted(set(bank['routes'])):
        ids=[i for i,e in enumerate(episodes) if e['route']==side]
        groups[side]=dict(n=len(ids),actual_return_gamma099=stats([episodes[i]['rtg'][0] for i in ids]),undiscounted=stats([episodes[i]['rewards'].sum() for i in ids]),success=sum(episodes[i]['goal']>0 for i in ids))
    np.savez_compressed(args.output/'reference_state_action_bank.npz',**bank)
    summary=dict(task=args.task,training_source=cfg['source_commit'],script_sha256=digest(__file__),checkpoint_sha256=hashes,config=cfg,
        reference_groups=groups,reference_reproduction_max_xy_error=errors,cross_checkpoint_q=cross,
        interpretation='Route labels describe realized whole rollouts, not causal one-action modes. t0 shares exactly the same full initial state; later t compares different visited states. Cross checkpoint comparison keeps every reference state/action fixed.',
        timeout='Actual return-to-go ends at timeout; critic bootstraps time limits, so late-horizon raw return differences are not automatically Q errors.')
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(phase='q_bank_done',groups=groups,cross_t0={str(s):{k:v['0'] for k,v in cross[str(s)].items()} for s in steps})),flush=True)
    # Exact one training-teacher candidate cloud at the identical initial state.
    load(250112);env=vector(args.task,64,seed=87231,asynchronous=False,fixed=True,reward_profile=cfg['reward_profile'],random_init=False)
    obs=env.reset();common=env.envs[0].initial
    first=jnp.asarray(obs[:1]);_,latent_key,proposal_key,_=jax.random.split(jax.random.PRNGKey(887231),4);zkey,_=jax.random.split(latent_key)
    z=jax.random.normal(zkey,(1,64,8));expanded=jnp.repeat(first,64,axis=0)
    mu,ls=p.actor_state.apply_fn({'params':p.actor_state.params},expanded,z.reshape(64,8));mu=mu.reshape(1,64,8);ls=ls.reshape(1,64,8)
    proposal=ConditionalGaussianProposal(mu,ls,float(native.alg.actor.proposal_std))
    candidates,u,component=proposal.sample(proposal_key,1,'exact');candidates=np.asarray(candidates[0]);logq=np.asarray(proposal.log_prob(u)[0])
    q=qall(p.qf_state,np.repeat(np.asarray(first),64,axis=0),candidates);qmean=q.mean(0)
    weights=np.asarray(jax.nn.softmax(jnp.asarray(qmean/float(native.alg.actor.temperature)-float(native.alg.actor.density_beta)*logq)))
    qweights=np.asarray(jax.nn.softmax(jnp.asarray(qmean/float(native.alg.actor.temperature))))
    @jax.jit
    def common_noise_actions(state,o,key):
        return jax.vmap(lambda x:p.sample_action(state,x[None],key,deterministic=False,sample_conditional_noise=True)[0])(o)
    outcomes=[];returns=[];boots=[];successes=[]
    for rep in range(args.continuations):
        obs=env.reset()
        for i,e in enumerate(env.envs):e.initial=common;obs[i]=e.restore(common)
        active=np.ones(64,bool);xy=[[o[:2].copy()] for o in obs];g=np.zeros(64);goal=np.zeros(64,int);last=np.zeros((64,29),np.float32);terminated=np.zeros(64,bool);lens=np.zeros(64,int);key=jax.random.PRNGKey(900001+rep)
        for t in range(700):
            key,sub=jax.random.split(key)
            action=candidates if t==0 else np.asarray(common_noise_actions(p.actor_state,jnp.asarray(obs),sub))
            nxt,reward,done,infos=env.step(action);final,terminal=transition(nxt,done,infos)
            for i in np.flatnonzero(active):
                xy[i].append(final[i,:2].copy());g[i]+=.99**t*float(reward[i]);lens[i]+=1;last[i]=final[i]
                if done[i]:active[i]=False;goal[i]=int(infos[i].get('success',0));terminated[i]=bool(terminal[i])
            obs=nxt
            if not active.any():break
        # Time-limit correction shown separately; this is critic-bootstrap, not independent MC evidence.
        many=np.repeat(last,64,axis=0);a=np.asarray(p.sample_action(p.actor_state,jnp.asarray(many),jax.random.PRNGKey(911000+rep),deterministic=False,sample_conditional_noise=True))
        v=qall(p.qf_state,many,a).mean(0).reshape(64,64).mean(1);boot=g+np.where(terminated,0,.99**lens*v)
        outcomes.append([route(args.task,np.asarray(x)) for x in xy]);returns.append(g);boots.append(boot);successes.append(goal)
        print(json.dumps(dict(phase='forced_first_action_continuation',rep=rep,seconds=time.monotonic()-started,counts={r:outcomes[-1].count(r) for r in set(outcomes[-1])})),flush=True)
    env.close();outcomes=np.asarray(outcomes);returns=np.asarray(returns);boots=np.asarray(boots)
    teacher={}
    for side in sorted(set(outcomes.flat)):
        probability=(outcomes==side).mean(0);teacher[side]=dict(proposal_probability=float(probability.mean()),teacher_probability=float(weights@probability),q_only_probability=float(qweights@probability),rep_proposal=((outcomes==side).mean(1)).tolist(),rep_teacher=((outcomes==side)@weights).tolist())
    summary['teacher_probe']=dict(seed=887231,candidate_count=64,continuations=args.continuations,common_continuation_randomness=True,route_mass=teacher,
        q_mean_range=[float(qmean.min()),float(qmean.max())],q_mean=stats(qmean),q_twin_gap=stats(abs(q[0]-q[1])),ess=float(1/(weights**2).sum()),max_weight=float(weights.max()),
        actual_mc=stats(returns.mean(0)),bootstrap_mc=stats(boots.mean(0)),note='One candidate cloud; paired continuation seeds. Only first action differs. A first torque does not uniquely choose a long route; no torque-sign labels used.')
    np.savez_compressed(args.output/'teacher_probe.npz',candidates=candidates,q=q,logq=logq,weights=weights,q_only_weights=qweights,outcomes=outcomes,returns=returns,bootstrap_returns=boots,successes=successes,mu=np.asarray(mu),logstd=np.asarray(ls))
    assert hashlib.sha256(fs.to_bytes(saved)).hexdigest()==initial_hash
    assert all(digest(path)==h for path,h in hashes.items())
    summary['verification']=dict(checkpoints_unchanged=True,restored_states_unchanged=True,no_training_updates=True,seconds=time.monotonic()-started)
    write(args.output/'summary.json',summary);write(args.output/'result.json',dict(completed=True,summary='summary.json'))
    print(json.dumps(dict(phase='completed',teacher=summary['teacher_probe'])),flush=True)

if __name__=='__main__':main()
