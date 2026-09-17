"""Validation jobs are separate from the 448 comparison trajectories."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.optimize import linprog
from scipy.special import logsumexp
from .config import METHODS,SIZES,tasks,write_json
from .problems import schedule,analytic_q,analytic_cdf,reference,GRID,PositionChoice,reward
from .models import actor_state,critic_state,td_update,sample_action,critic_model
from .actor import engine
from .exact1d import monotone_plan
from .diagnostics import interpolation_check
from .io import verify_source,sha
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.distillation import direct_gmm_nll,conditional_ot_nll

def tree_error(a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))

def identities():
    records=[]
    for stage,t,k in [('mass',20000,3),('mass',20001,3),('mass',25001,3),('mass',30001,3),
                      ('split',20000,3),('split',22000,6),('split',27000,3)]:
        p=schedule(stage,t);ref=reference(analytic_q(GRID,p),p)
        assert len(ref['peaks'])==k,(stage,t,ref['peaks'])
        assert abs(np.trapz(ref['pdf'],GRID)-1)<1e-10
        np.testing.assert_allclose(analytic_cdf([-1,1],p),[0,1],atol=1e-14)
        records.append(dict(test='schedule_peaks_and_normalization',stage=stage,step=t,peaks=k))
    assert not np.array_equal(schedule('mass',20000),schedule('mass',20001))
    assert np.array_equal(schedule('prefix',0),schedule('split',27000))
    env=PositionChoice()
    for i in range(200):
        prev=env.state;sp,r,term,trunc=env.step(.6)
        assert not term and trunc==(i==199) and sp==.6
        np.testing.assert_allclose(r,reward(prev,.6))
    assert env.state==0 and env.episode_step==0
    # Independent LP checks, including ties and vanishing teacher masses.
    rng=np.random.default_rng(9)
    for trial in range(5):
        x=rng.normal(size=4);b=rng.normal(size=9);w=rng.dirichlet(np.ones(9))
        if trial==0:x[:]=0
        if trial==1:w[0]=0;w/=w.sum()
        C=(x[:,None]-b[None])**2;P=monotone_plan(x,b,w)
        A=np.zeros((13,36))
        for i in range(4):A[i,i*9:(i+1)*9]=1
        for j in range(9):A[4+j,j::9]=1
        sol=linprog(C.ravel(),A_eq=A,b_eq=np.r_[np.full(4,.25),w],bounds=(0,None),method='highs')
        assert sol.success
        gap=abs((P*C).sum()-sol.fun)
        assert gap<2e-6 and abs(P.sum(0)-w).sum()<2e-7 and abs(P.sum(1)-.25).sum()<2e-7
        records.append(dict(test='exact_vs_independent_LP',cost_gap=gap))
    # Marginal likelihood and independent output-level gradient.
    mu=rng.normal(size=(1,3,1)).astype('float32');ls=rng.uniform(-1,0,size=mu.shape).astype('float32')
    u=rng.normal(size=(1,7,1)).astype('float32');w=rng.dirichlet(np.ones(7),size=1).astype('float32')
    ell=(-.5*((u[:,None]-mu[:,:,None])/np.exp(ls[:,:,None]))**2-ls[:,:,None]-.5*np.log(2*np.pi)).sum(-1)
    expected=-(w*(logsumexp(ell,axis=1)-np.log(3))).sum()
    value=direct_gmm_nll(jnp.array(mu),jnp.array(ls),jnp.array(u),jnp.array(w))[0]
    np.testing.assert_allclose(value,expected,rtol=2e-6)
    gm,gl,gu,gw=jax.grad(lambda m,l,v,w:direct_gmm_nll(m,l,v,w)[0],argnums=(0,1,2,3))(*map(jnp.array,(mu,ls,u,w)))
    A=w[:,None]*np.exp(ell-logsumexp(ell,axis=1,keepdims=True));d=mu[:,:,None]-u[:,None]
    np.testing.assert_allclose(gm,(A[...,None]*d).sum(2)*np.exp(-2*ls),atol=2e-5)
    np.testing.assert_allclose(gl,(A[...,None]*(1-d*d*np.exp(-2*ls[:,:,None]))).sum(2),atol=2e-5)
    assert not np.any(gu) and not np.any(gw)
    records.append(dict(test='GMM_joint_likelihood_gradient_stop_teacher',passed=True))
    return records

def parity():
    records=[];obs=jnp.array([[.2]],jnp.float32);critic=critic_state(1)
    for method in ('sinkhorn_learned','gmm_learned'):
        state=actor_state(method,0);key=jax.random.PRNGKey(718)
        f=engine(method,16,64,'critic');ours=f['step'](state,obs,key,critic.params)
        v5=OptiQDIME.update_actor(state,critic,obs,key,jnp.array([-3600.]),16,4,'exact',.05,.5,
            False,True,1.,False,16.,257,.25,.1,100,'mean','argmax',True,False,
            'direct_gmm_nll' if method.startswith('gmm') else 'conditional_ot_nll','conditional_mixture',0.,False,'mean')
        err=tree_error(ours[0].params,v5[0].params)
        assert err<3e-5,(method,err)
        np.testing.assert_allclose(ours[3],v5[1],rtol=2e-5,atol=1e-6)
        np.testing.assert_array_equal(ours[1],v5[2])
        records.append(dict(test='actual_v5_actor_update',method=method,parameter_linf=err))
    # Same current actor, min target Q, no entropy; verify loss independently.
    state=actor_state('exact_fixed01',2);k=jax.random.PRNGKey(17)
    batch={'s':jnp.array([[0.],[.5]]),'a':jnp.array([[.1],[-.2]]),'sp':jnp.array([[.1],[-.2]]),'r':jnp.array([1.,-.1])}
    _,ak,*_=jax.random.split(k,6);a=sample_action(state,batch['sp'],ak,deterministic=False)
    qt=critic.apply_fn({'params':critic.target_params},batch['sp'],a,train=False)[...,0]
    qc=critic.apply_fn({'params':critic.params},batch['s'],batch['a'],train=False)[...,0]
    expected=np.square(np.asarray(qc)-(np.asarray(batch['r'])+.99*np.asarray(qt).min(0))[None]).mean(1).sum()
    new,metrics,_=td_update(state,critic,batch,k)
    np.testing.assert_allclose(metrics['critic_loss'],expected,rtol=2e-6)
    assert metrics['ent_coef']==0
    expected_target=jax.tree_util.tree_map(lambda p,t:.005*p+.995*t,new.params,critic.target_params)
    assert tree_error(new.target_params,expected_target)<1e-7
    records.append(dict(test='v5_reward_only_TD_min_target_and_Polyak',passed=True))
    # GMM must not invoke either OT solver, even during a fresh trace.
    import experiment.actor as mod
    old_s,old_p=mod.sinkhorn,mod.monotone_plan
    def forbidden(*args,**kwargs):raise AssertionError('GMM called OT')
    mod.sinkhorn=mod.monotone_plan=forbidden
    try:engine('gmm_learned',17,68,'analytic')['step'](actor_state('gmm_learned',0),jnp.zeros((1,1)),k,jnp.array(schedule('mass',25001)))
    finally:mod.sinkhorn,mod.monotone_plan=old_s,old_p
    records.append(dict(test='GMM_no_OT',passed=True))
    return records

def integration(root):
    from .run import run
    ts=tasks();find=lambda stage:next(t.copy() for t in ts if t['stage']==stage and t['method']=='sinkhorn_learned' and t['n']==16 and t['seed']==0)
    prefix=find('prefix');run(root,prefix,True,2,32)
    ph=sha(root/'validation_runs'/prefix['name']/'checkpoint.msgpack')
    for stage in ['mass','split']:run(root,find(stage),True,5,32)
    assert sha(root/'validation_runs'/prefix['name']/'checkpoint.msgpack')==ph
    source=find('source');run(root,source,True,6,32)
    run(root,find('replay'),True,6,32)
    closed=find('closed');a=closed.copy();a['name']='resume_interrupted';b=closed.copy();b['name']='resume_uninterrupted'
    try:run(root,a,True,6,32,interrupt_at=3)
    except SystemExit as e:assert e.code==75
    run(root,a,True,6,32);run(root,b,True,6,32)
    for filename in ['final_actor.msgpack','final_critic.msgpack']:
        assert sha(root/'validation_runs'/a['name']/filename)==sha(root/'validation_runs'/b['name']/filename),filename
    aa=serialization.msgpack_restore((root/'validation_runs'/a['name']/'checkpoint.msgpack').read_bytes())
    bb=serialization.msgpack_restore((root/'validation_runs'/b['name']/'checkpoint.msgpack').read_bytes())
    np.testing.assert_array_equal(aa['key'],bb['key']);np.testing.assert_array_equal(aa['collectkey'],bb['collectkey'])
    assert aa['replay']['rng']==bb['replay']['rng'] and aa['env_state']==bb['env_state']
    q=np.load(root/'validation_runs'/source['name']/'qstream/000001.npy')
    assert q.shape==(6,8193)
    return [dict(test='all_stages_prefix_immutable_Qstream_resume_exact',passed=True)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--size-index',type=int,required=True)
    a=p.parse_args();root=Path(a.root);code=verify_source(root);assert jax.default_backend()=='gpu'
    records=identities() if a.size_index==0 else []
    if a.size_index==0:records+=parity()
    n,m=SIZES[a.size_index];timings=[]
    for method in METHODS:
        state=actor_state(method,0);f=engine(method,n,m,'analytic');key=jax.random.PRNGKey(16);times=[]
        for i in range(6):
            initial=state;tick=time.perf_counter()
            state,key,t,v,gn=f['step'](state,jnp.zeros((1,1)),key,jnp.array(schedule('split',22000) if i%2 else schedule('mass',25001)))
            stats=jax.device_get(f['scalars'](state.params,t,v,gn));times.append(time.perf_counter()-tick)
            assert all(np.isfinite(v).all() for v in stats.values())
            if METHODS[method][1] is not None:
                mu,ls=f['heads'](state.params,jnp.zeros((1,1)),jnp.zeros((1,n,1)))
                np.testing.assert_allclose(np.exp(ls),METHODS[method][1],rtol=2e-7)
                assert tree_error(state.params['log_std'],initial.params['log_std'])==0
            if method.startswith('exact'):assert stats['ot_row_l1']<2e-6 and stats['ot_col_l1']<2e-6
        timings.append(dict(method=method,n=n,m=m,seconds_per_update=float(np.mean(times[2:])),last={k:float(v) for k,v in stats.items()}))
        # Test full sampling/assignment export at every requested size/method.
        from .diagnostics import evaluate
        task=dict(method=method,seed=0);out=root/'validation_runs'/f'shape_{n}_{method}';out.mkdir(parents=True,exist_ok=True)
        evaluate(out,task,1,state,f,jnp.array(schedule('split',22000)),'analytic')
    if a.size_index==0:records+=integration(root)
    write_json(root/f'VALIDATION_{a.size_index}.json',dict(passed=True,source_code_id=code,
        records=records,timings=timings,device=str(jax.devices()[0]),finished_unix=time.time()))
    print('VALIDATION PASSED',a.size_index,flush=True)

if __name__=='__main__':main()
