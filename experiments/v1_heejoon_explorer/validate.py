"""Role separation, exact TD formula, legacy actor parity, EMA and full resume."""
import argparse
import tempfile
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from optiq_dime.algorithm import OptiQDIME
from .algorithm import ExplorerOptiQ, continuation_target
from .run import config, make_model, verify_source, write
from . import checkpoint


def difference(a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
               for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def actor_fn(variables, obs, z):
    return jnp.broadcast_to(variables['params']['Dense_0']['bias'], (obs.shape[0],1))


def critic_fn(variables, obs, action, train=False, mutable=None, rngs=None):
    p=variables['params'];value=p[:,0,None]+p[:,1,None]*action[None,:,0]
    value=value[...,None]
    return (value, {'batch_stats':{}}) if mutable else value


def td_tests():
    # Evaluator action +0.4; target critics are (1+a, 2-a), live critics differ.
    actor=TrainState.create(apply_fn=actor_fn,params={'Dense_0':{'bias':jnp.array([.4])}},tx=optax.sgd(.1))
    critic=RLTrainState.create(apply_fn=critic_fn,params=jnp.array([[10.,3.],[20.,4.]]),
                              target_params=jnp.array([[1.,1.],[2.,-1.]]),
                              batch_stats={},target_batch_stats={},tx=optax.sgd(.1))
    obs=jnp.zeros((3,1));rew=jnp.array([0.,1.,2.]);done=jnp.array([0.,0.,1.]);key=jax.random.PRNGKey(9)
    target,actions=continuation_target(actor,critic,obs,rew,done,.99,key)
    np.testing.assert_allclose(actions,.4)
    np.testing.assert_allclose(target,np.asarray(rew)+.99*(1-np.asarray(done))*1.4,rtol=1e-6)
    grad=jax.grad(lambda params:continuation_target(actor.replace(params=params),critic,obs,rew,done,.99,key)[0].sum())(actor.params)
    assert all(np.count_nonzero(np.asarray(v))==0 for v in jax.tree_util.tree_leaves(grad))
    grad=jax.grad(lambda params:continuation_target(actor,critic.replace(target_params=params),obs,rew,done,.99,key)[0].sum())(critic.target_params)
    assert np.count_nonzero(np.asarray(grad))==0
    return dict(exact_twin_min_target=True,terminal_mask=True,td_stop_gradient=True)


def validate_environment(env,out,commit):
    cfg=config(env,0,str(out),smoke=True);model,callbacks=make_model(cfg)
    try:
        policy=model.policy;obs=np.zeros((1,*model.observation_space.shape),np.float32)
        original=policy.actor_state
        p=jax.tree_util.tree_map(lambda x:jnp.zeros_like(x),original.params)
        key=f'Dense_{len(p)-1}';p[key]['bias']=jnp.full_like(p[key]['bias'],.4)
        policy.actor_state=original.replace(params=p)
        q=jax.tree_util.tree_map(lambda x:jnp.zeros_like(x),p);q[key]['bias']=jnp.full_like(q[key]['bias'],-.4)
        policy.target_actor_state=policy.target_actor_state.replace(params=q)
        expected_exp=policy.unscale_action(np.full((1,*model.action_space.shape),.4))
        expected_eval=policy.unscale_action(np.full((1,*model.action_space.shape),-.4))
        np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],expected_exp,atol=1e-7)
        policy.evaluation_only=True;before=np.asarray(jax.random.key_data(policy.key)).copy()
        np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],expected_eval,atol=1e-7)
        np.testing.assert_array_equal(before,jax.random.key_data(policy.key));policy.evaluation_only=False
        # EMA touches only evaluator parameters; it is never optimizer-trained.
        updated=ExplorerOptiQ.soft_update_target_actor(.005,policy.actor_state,policy.target_actor_state)
        np.testing.assert_allclose(updated.params[key]['bias'],.995*(-.4)+.005*.4,rtol=1e-6)
        assert int(updated.step)==0
        policy.actor_state=original;policy.target_actor_state=policy.target_actor_state.replace(params=original.params)
        assert ExplorerOptiQ.update_actor is OptiQDIME.update_actor
        # Every actor update must be followed by exactly one EMA of its new params.
        captures=[];previous=policy.target_actor_state.params
        def observe(m):
            nonlocal previous
            expected=optax.incremental_update(m.policy.actor_state.params,previous,.005)
            err=difference(expected,m.policy.target_actor_state.params)
            assert err<2e-7,err
            assert int(m.policy.target_actor_state.step)==0 and not m.policy.evaluation_only
            captures.append(err);previous=m.policy.target_actor_state.params
        model.checkpoint_hook=observe
        model.learn(total_timesteps=80,callback=callbacks,progress_bar=False,log_interval=1)
        assert len(captures)==48 and model.replay_buffer.pos==80
        path=out/'validation_resume.zip'
        checkpoint.save(model,callbacks,path,commit)
        expected=serialization.to_bytes(checkpoint.states(model))
        expected_keys=[np.asarray(jax.random.key_data(k)) for k in [model.key,policy.key,policy.eval_key]]
        expected_action=model.predict(model._last_obs)[0]
        next_obs,reward,done,info=model.get_env().step(expected_action)
        checkpoint.load(model,callbacks,path,commit)
        assert serialization.to_bytes(checkpoint.states(model))==expected
        for a,b in zip(expected_keys,[model.key,policy.key,policy.eval_key]):np.testing.assert_array_equal(a,jax.random.key_data(b))
        resumed_action=model.predict(model._last_obs)[0]
        np.testing.assert_array_equal(expected_action,resumed_action)
        resumed_obs,resumed_reward,resumed_done,_=model.get_env().step(resumed_action)
        np.testing.assert_allclose(next_obs,resumed_obs,rtol=0,atol=0)
        np.testing.assert_allclose(reward,resumed_reward,rtol=0,atol=0)
        np.testing.assert_array_equal(done,resumed_done)
        assert len(callbacks.callbacks[0].evaluations_results)>=2
        return dict(environment=env,trained_updates=len(captures),max_ema_error=max(captures),
                    evaluator_gradient_steps=int(policy.target_actor_state.step),role_routing=True,
                    evaluation_does_not_consume_explorer_rng=True,legacy_actor_inherited=True,
                    full_checkpoint_roundtrip=True,next_action_and_simulator_exact=True)
    finally:
        model.get_env().close();model.logger.close()
        for cb in callbacks.callbacks:
            if hasattr(cb,'eval_env'):cb.eval_env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=verify_source();assert jax.default_backend()=='gpu' and len(jax.devices())==1
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(commit=manifest['commit'],source_code_id=manifest['source_code_id'],**td_tests(),environments=[])
    for env in ['ant','humanoid','halfcheetah']:
        out=args.output/env;out.mkdir(exist_ok=True)
        result['environments'].append(validate_environment(env,out,manifest['commit']))
        write(args.output/'VALIDATION_PROGRESS.json',result)
    result['passed']=True;write(args.output/'VALIDATION_PASSED.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
