"""Replay identical evaluation prefixes; inspect changing full physical states."""
import json
import numpy as np


def diagnose(learner,reference,output):
    import jax.numpy as jnp
    from .envs import vector, make_one
    from .learners import evaluation_rng
    data=np.load(reference/'evaluations/0000100000/native-fixed/rollouts.npz')
    xy=data['xy']
    selected=np.flatnonzero((xy[:20,10,1]>0)&(xy[:20,50,1]<0))[:6]
    assert len(selected)>0
    snapshots={}
    with evaluation_rng(learner,800000):
        env=vector('v1',20,seed=87231,asynchronous=False,fixed=True,
                   reward_profile=learner.reward_profile,random_init=False)
        obs=env.reset();common=env.envs[0].initial
        for i,e in enumerate(env.envs):
            e.initial=common;obs[i]=e.restore(common)
        for t in range(51):
            np.testing.assert_array_equal(obs[:,:2],xy[:20,t])
            if t in (10,20,25):
                for i in selected:snapshots[(int(i),t)]=(env.envs[i].state(),obs[i].copy())
            if t<50:obs,_,done,_=env.step(learner.act(obs,'native'));assert not done.any()
        env.close()
    probe=make_one('v1',1,reward_profile=learner.reward_profile,random_init=False)
    probe.reset();rows=[]
    for index,((episode,t),(state,obs)) in enumerate(snapshots.items()):
        with evaluation_rng(learner,92000+index):
            mu=learner.act(np.repeat(obs[None],128,axis=0),'native')
            full=learner.act(np.repeat(obs[None],128,axis=0),'policy')
        uniform=np.random.default_rng(93000+index).uniform(-1,1,(128,8)).astype(np.float32)
        actions=np.concatenate([mu,full,uniform])
        p=learner.model.policy
        qs=np.asarray(p.qf_state.apply_fn({'params':p.qf_state.params,'batch_stats':p.qf_state.batch_stats},
             jnp.asarray(np.repeat(obs[None],len(actions),axis=0)),jnp.asarray(actions),train=False))[...,0].T
        delta1=[];delta10=[]
        for action in actions:
            probe.restore(state)
            for k in range(10):
                nxt,_,done,_=probe.step(action)
                if k==0:delta1.append(nxt[:2]-obs[:2])
                if done:break
            delta10.append(nxt[:2]-obs[:2])
        delta1=np.array(delta1);delta10=np.array(delta10)
        row=dict(episode=episode,step=t,xy=obs[:2].tolist(),velocity_xy=np.asarray(state['qvel'][:2]).tolist())
        for name,sl in [('mu',slice(0,128)),('full',slice(128,256)),('uniform',slice(256,384))]:
            dy=delta10[sl,1];q=qs[sl].mean(axis=1);up=dy>.001;down=dy<-.001
            top=np.argsort(q)[-13:]
            row[name]=dict(up_fraction=float(up.mean()),down_fraction=float(down.mean()),
                one_step_up_fraction=float((delta1[sl,1]>0).mean()),
                q_up=float(q[up].mean()) if up.any() else None,
                q_down=float(q[down].mean()) if down.any() else None,
                top_q_up_fraction=float(up[top].mean()),q_std=float(q.std()))
        # Each candidate is applied ONCE, then the real mu-only policy resumes.
        # Compare highest-Q mu candidates with random mu candidates under paired noise.
        qa=qs[:128].mean(axis=1)
        chosen=np.r_[np.argsort(qa)[-8:],np.random.default_rng(94000+index).choice(128,8,replace=False)]
        follow_dy=[]
        for candidate in chosen:
            current=probe.restore(state)
            with evaluation_rng(learner,95000+index):
                for k in range(25):
                    action=actions[candidate] if k==0 else learner.act(current[None],'native')[0]
                    current,_,done,_=probe.step(action)
                    if done:break
            follow_dy.append(float(current[1]-obs[1]))
        row['one_action_then_policy']=dict(top_q_up_fraction=float((np.array(follow_dy[:8])>0).mean()),
            random_up_fraction=float((np.array(follow_dy[8:])>0).mean()),
            top_q_mean_dy=float(np.mean(follow_dy[:8])),random_mean_dy=float(np.mean(follow_dy[8:])))
        np.savez_compressed(output/f'episode{episode}_t{t}.npz',obs=obs,actions=actions,q=qs,delta1=delta1,delta10=delta10,
                            chosen=chosen,follow_dy=follow_dy)
        rows.append(row)
        (output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(snapshots))))
        print(json.dumps(row),flush=True)
    probe.close()
    return dict(selected_episodes=selected.tolist(),selection='first six of first20 episodes: y10>0 and y50<0',
                replay_prefix_exact=True,conditional_noise_in_evaluation=False,rows=rows,
                caveats=['10-step constant action direction is a local probe, not a route-value label',
                         'Uniform candidates may be out of distribution',
                         'Critic trains with noisy policy targets; continuation probes here use mu-only'])
