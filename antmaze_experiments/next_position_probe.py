"""20,000 independent one-step physical interventions from identical s0."""
import json
import numpy as np


def diagnose(learner,output):
    import jax
    import jax.numpy as jnp
    from .envs import make_one
    env=make_one('v1',0,reward_profile=learner.reward_profile,random_init=False)
    obs=env.reset();initial=env.state();n=20000
    z=jax.random.normal(jax.random.PRNGKey(98000),(n,8))
    p=learner.model.policy
    mu,ls=p.actor_state.apply_fn({'params':p.actor_state.params},jnp.asarray(np.repeat(obs[None],n,axis=0)),z)
    actions=np.asarray(mu);positions=[];rewards=[]
    def single(action):
        env.reset()  # Clear simulator integration/contact/warmstart state as well.
        restored=env.restore(initial)
        np.testing.assert_array_equal(restored,obs)
        nxt,r,done,info=env.step(action)
        return nxt.copy(),float(r)
    for i,action in enumerate(actions):
        nxt,r=single(action);positions.append(nxt[:2]);rewards.append(r)
        if (i+1)%2000==0:
            print(json.dumps(dict(samples=i+1,total=n)),flush=True)
    positions=np.asarray(positions)
    # Check repeated interventions are exactly reproducible, independent of order.
    for i in (0,17,901,19999):
        repeat,_=single(actions[i]);np.testing.assert_array_equal(repeat[:2],positions[i])
    dt=float(env.physics_env.dt)
    env.close()
    np.savez_compressed(output/'next_positions.npz',z=np.asarray(z),mu=actions,log_std=np.asarray(ls),
                        initial_obs=obs,positions=positions,rewards=np.asarray(rewards),dt=dt)
    return dict(samples=n,env_steps_per_sample=1,physical_dt_seconds=dt,
                conditional_noise=False,initial_xy=obs[:2].tolist(),
                positive_y=int((positions[:,1]>0).sum()),negative_y=int((positions[:,1]<0).sum()),
                positive_x=int((positions[:,0]>0).sum()),negative_x=int((positions[:,0]<0).sum()),
                xy_mean=positions.mean(0).tolist(),xy_median=np.median(positions,axis=0).tolist(),
                xy_min=positions.min(0).tolist(),xy_max=positions.max(0).tolist(),
                exact_repeated_intervention_verified=True,
                note='Each z produces mu; reset identical full physical state before one env.step. Not trajectory rollouts.')
