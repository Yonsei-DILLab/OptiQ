"""Target min/MC order, gradient isolation, Adam skip, routing and full resume."""
import argparse,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from optiq_dime.algorithm import OptiQDIME
from experiments.v1_heejoon_explorer.validate import td_tests,difference
from .algorithm import ExplorerOptiQ,evaluator_teacher,evaluator_loss,fit_evaluator,latent_actions
from .run import config,make_model,verify_source,write
from . import checkpoint


def actor_fn(variables,obs,z):
    p=variables['params']['Dense_0'];return p['bias'][None]+p['kernel'][None]*z+p['state_gain'][None]*obs[:,:1]


def critic_fn(variables,obs,action,train=False,mutable=None,rngs=None):
    p=variables['params'];values=(p[:,0,None]+p[:,1,None]*action[None,:,0])[...,None]
    return (values,{'batch_stats':{}}) if mutable else values


def toy_actor(bias=0.,kernel=0.,state_gain=0.):
    return TrainState.create(apply_fn=actor_fn,params={'Dense_0':dict(bias=jnp.array([bias]),kernel=jnp.array([kernel]),state_gain=jnp.array([state_gain]))},tx=optax.adam(.01))


def same_state(a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)


def unit_tests():
    critic=RLTrainState.create(apply_fn=critic_fn,params=jnp.array([[100.,-10.],[200.,-20.]]),target_params=jnp.array([[1.,1.],[1.25,1.]]),batch_stats={},target_batch_stats={},tx=optax.adam(.1))
    evaluator=toy_actor();explorer=toy_actor(bias=.25,state_gain=.25);obs=jnp.array([[-1.],[0.],[1.]]);key=jax.random.PRNGKey(2)
    t=evaluator_teacher(evaluator,explorer,critic,obs,key)
    np.testing.assert_allclose(t['advantage'],[-.25,0.,.25],atol=1e-6)
    np.testing.assert_array_equal(t['accepted'],[0,0,1]);np.testing.assert_allclose(t['q_eval'],1.25,atol=1e-6)
    np.testing.assert_array_equal(t['weights'],[0.,0.,.25])
    np.testing.assert_allclose(evaluator_loss(evaluator.params,evaluator,t),.0625,rtol=1e-6)
    new,loss,gn=fit_evaluator(evaluator,t);assert int(new.step)==1 and float(gn)>0
    # Unequal positive advantages must change regression relative to v2.
    weighted=evaluator_teacher(evaluator,toy_actor(bias=.5,state_gain=.5),critic,jnp.array([[0.],[1.]]),key)
    np.testing.assert_array_equal(weighted['weights'],[.25,.75])
    value=evaluator_loss(evaluator.params,evaluator,weighted)
    np.testing.assert_allclose(value,.40625,rtol=1e-6)
    np.testing.assert_allclose(evaluator_loss(evaluator.params,evaluator,{**weighted,'weights':2*weighted['weights']}),2*value,rtol=1e-6)
    grads=jax.grad(evaluator_loss)(evaluator.params,evaluator,weighted)
    np.testing.assert_allclose(grads['Dense_0']['bias'],[-.875],rtol=1e-6)
    # Graph-level isolation: target critic and explorer gradients must be exactly zero.
    def full_loss(ep,xp,qp):
        ev=evaluator.replace(params=ep);ex=explorer.replace(params=xp);cr=critic.replace(target_params=qp)
        teacher=evaluator_teacher(ev,ex,cr,obs,key)
        return evaluator_loss(ep,ev,teacher)
    ge,gx,gq=jax.grad(full_loss,argnums=(0,1,2))(evaluator.params,explorer.params,critic.target_params)
    assert any(np.count_nonzero(x) for x in jax.tree_util.tree_leaves(ge))
    assert all(np.count_nonzero(x)==0 for x in jax.tree_util.tree_leaves(gx)) and np.count_nonzero(gq)==0
    # Live Q cannot affect the gate.
    other=evaluator_teacher(evaluator,explorer,critic.replace(params=-critic.params),obs,key)
    np.testing.assert_array_equal(t['advantage'],other['advantage'])
    # Momentum already nonzero, then no positive samples: weights AND optimizer unchanged.
    rejected={**t,'accepted':jnp.zeros_like(t['accepted'])}
    skipped,vl,_=fit_evaluator(new,rejected);same_state(new,skipped);assert float(vl)==0.
    # E[max Q1,Q2] must be used, NOT max(E Q1,E Q2).
    ev=toy_actor(kernel=.4);ex=toy_actor();cr=critic.replace(target_params=jnp.array([[0.,1.],[0.,-1.]]))
    tt=evaluator_teacher(ev,ex,cr,jnp.zeros((4,1)),key)
    actions=np.asarray(tt['baseline_actions'])[...,0]
    np.testing.assert_allclose(tt['q_eval'],np.abs(actions).mean(1),rtol=1e-6)
    assert np.all(np.asarray(tt['q_eval'])>np.abs(actions.mean(1))+1e-3)
    assert np.all(np.asarray(tt['accepted'])==0)
    assert np.all(np.asarray(tt['q_exp']-tt['q_eval_min'])>0)
    assert np.all(np.asarray(tt['advantage'])<=np.asarray(tt['q_exp']-tt['q_eval_min']))
    # Positive advantage does not connect critic differentiation to the prediction.
    # Same-latent MSE is explicit in t['z'], not an independent evaluator draw.
    predicted=latent_actions(ev,jnp.zeros((4,1)),tt['z']);assert predicted.shape==tt['actions'].shape
    assert ExplorerOptiQ.update_actor is OptiQDIME.update_actor
    same_state(ExplorerOptiQ.soft_update_target_actor(.9,explorer,evaluator),evaluator)
    return dict(strict_positive_gate=True,target_not_live_critic=True,max_before_mc_mean=True,
        min_max_advantage=True,raw_advantage_weights=True,weighted_loss_and_gradient_exact=True,
        counterfactual_minmin_acceptance_superset=True,
        accepted_count_loss_normalization=True,explorer_and_target_critic_gradients_zero=True,
        evaluator_prediction_gradient_nonzero=True,no_positive_batch_skips_adam_momentum=True,
        no_parameter_ema=True,legacy_explorer_update_inherited=True,matched_latent_targets=True,**td_tests())


def validate_environment(env,out,commit):
    cfg=config(env,0,str(out),smoke=True)
    # CPU/GPU validation uses short replay batches; numerical gates above test exact formulas.
    cfg.alg.batch_size=8;cfg.alg.learning_starts=16;cfg.alg.actor.learning_starts=16
    cfg.eval_interval=32;cfg.diagnostic_interval=32;cfg.num_eval_episodes=1
    model,callbacks=make_model(cfg)
    try:
        policy=model.policy;obs=np.zeros((1,*model.observation_space.shape),np.float32)
        original_exp=policy.actor_state;original_eval=policy.target_actor_state
        p=jax.tree_util.tree_map(jnp.zeros_like,original_exp.params);last=f'Dense_{len(p)-1}';p[last]['bias']=jnp.full_like(p[last]['bias'],.4)
        policy.actor_state=original_exp.replace(params=p)
        q=jax.tree_util.tree_map(jnp.zeros_like,p);q[last]['bias']=jnp.full_like(q[last]['bias'],-.4)
        policy.target_actor_state=original_eval.replace(params=q)
        np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],policy.unscale_action(np.full((1,*model.action_space.shape),.4)),atol=1e-7)
        before=np.asarray(jax.random.key_data(policy.key)).copy();policy.evaluation_only=True
        np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],policy.unscale_action(np.full((1,*model.action_space.shape),-.4)),atol=1e-7)
        np.testing.assert_array_equal(before,jax.random.key_data(policy.key));policy.evaluation_only=False
        policy.actor_state=original_exp;policy.target_actor_state=original_eval
        # A strict min-max gate can reject every early random-critic sample.
        # Separately validate nonzero regression/Adam with a known teacher on
        # the actual network, without assigning this test state to the model.
        z=jax.random.normal(jax.random.PRNGKey(19),(3,*model.action_space.shape))
        test_obs=jnp.zeros((3,*model.observation_space.shape))
        target=jnp.clip(latent_actions(original_eval,test_obs,z)+.1,-1.,1.)
        fixture=dict(observations=test_obs,z=z,actions=target,
                     accepted=jnp.array([0.,1.,1.]),weights=jnp.array([0.,.25,.75]))
        test_state,test_loss,test_grad=fit_evaluator(original_eval,fixture)
        assert int(test_state.step)==int(original_eval.step)+1 and float(test_grad)>0
        previous=policy.target_actor_state;updates=[]
        def observe(m):
            nonlocal previous
            current=m.policy.target_actor_state
            assert int(current.step)-int(previous.step) in (0,1)
            if int(current.step)==int(previous.step):same_state(current,previous)
            assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(jax.device_get(current)))
            updates.append(int(current.step));previous=current
        model.checkpoint_hook=observe
        started=time.monotonic();model.learn(total_timesteps=40,callback=callbacks,progress_bar=False,log_interval=1)
        assert len(updates)==24
        # A default-size batch proves shape integration at production B=256 too.
        old_steps=int(policy.target_actor_state.step);model.checkpoint_hook=None;model.train(batch_size=256,gradient_steps=1)
        assert int(policy.target_actor_state.step)-old_steps in (0,1)
        path=out/'validation_resume.zip';checkpoint.save(model,callbacks,path,commit)
        expected=serialization.to_bytes(checkpoint.states(model));expected_keys=[np.asarray(jax.random.key_data(k)) for k in [model.key,policy.key,policy.eval_key]]
        action=model.predict(model._last_obs)[0];next_obs,reward,done,_=model.get_env().step(action)
        checkpoint.load(model,callbacks,path,commit);assert serialization.to_bytes(checkpoint.states(model))==expected
        for a,b in zip(expected_keys,[model.key,policy.key,policy.eval_key]):np.testing.assert_array_equal(a,jax.random.key_data(b))
        restored_action=model.predict(model._last_obs)[0];np.testing.assert_array_equal(action,restored_action)
        robs,rrew,rdone,_=model.get_env().step(restored_action)
        np.testing.assert_array_equal(next_obs,robs);np.testing.assert_array_equal(reward,rrew);np.testing.assert_array_equal(done,rdone)
        # Restore once more for exact next *training* update including replay RNG/evaluator Adam.
        checkpoint.load(model,callbacks,path,commit);model.train(batch_size=8,gradient_steps=1)
        expected_after=checkpoint.states(model);expected_key=np.asarray(jax.random.key_data(model.key))
        checkpoint.load(model,callbacks,path,commit);model.train(batch_size=8,gradient_steps=1)
        same_state(expected_after,checkpoint.states(model));np.testing.assert_array_equal(expected_key,jax.random.key_data(model.key))
        return dict(env=env,small_batch_updates=24,default_batch256_update=True,evaluator_optimizer_steps=int(policy.target_actor_state.step),
            actual_network_weighted_adam_fixture_passed=True,zero_early_gate_acceptance_allowed=True,
            routing=True,evaluation_rng_independent=True,full_checkpoint_roundtrip=True,next_action_simulator_and_training_update_exact=True,elapsed_seconds=time.monotonic()-started)
    finally:
        model.get_env().close();model.logger.close()
        for cb in callbacks.callbacks:
            if hasattr(cb,'eval_env'):cb.eval_env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','gpu'],required=True)
    p.add_argument('--env',choices=['ant','humanoid','halfcheetah']);a=p.parse_args()
    manifest=verify_source();assert jax.default_backend()==a.device
    if a.device=='gpu':assert len(jax.devices())==1
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(commit=manifest['commit'],source_code_id=manifest['source_code_id'],device=a.device,checks=unit_tests(),environments=[])
    for env in [a.env] if a.env else ['ant','humanoid','halfcheetah']:
        out=a.output/env;out.mkdir(exist_ok=True);result['environments'].append(validate_environment(env,out,manifest['commit']))
        write(a.output/'VALIDATION_PROGRESS.json',result)
    result['passed']=True;write(a.output/'VALIDATION_PASSED.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
