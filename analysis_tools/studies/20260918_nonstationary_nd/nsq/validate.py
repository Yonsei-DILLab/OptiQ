"""Numerical, implementation, resume and throughput checks before main eligibility."""
import argparse,json,time,traceback,os
from pathlib import Path
import numpy as np
import scipy.optimize as so
import jax
import jax.numpy as jnp
from flax import serialization
from .config import METHODS,SIZES,write_json
from .problems import schedule,analytic_q,analytic_q_jax,analytic_reference,target_samples,reward
from .models import actor_state,critic_state,actor_apply
from .actor import engine
from .exact_nd import exact_plan
from .run import run,QStream
from .io import verify_source

def tree_diff(a,b):return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
def main(root,dim):
 root=Path(root);code=verify_source(root);assert jax.default_backend()=='gpu'
 record=dict(passed=False,source_code_id=code,dim=dim,commit=json.loads((root/'DEPLOYMENT.json').read_text())['commit'],gpu=str(jax.devices()[0]),checks=[],benchmarks=[])
 dest=root/f'VALIDATION_D{dim}.json'
 try:
  for family in ['double','tri']:
   for stage,t in [('prefix',0),('mass',20001),('mass',25001),('mass',35000)]+([('split',22000),('split',27000)] if family=='tri' else []):
    p=schedule(stage,t,family);ref=analytic_reference(p,dim)
    assert np.max(abs(ref['target_marginals'].sum(1)-1))<1e-10
    expected=2 if family=='double' else 6 if stage=='split' and t==22000 else 3
    assert len(ref['peaks'])==expected,(family,stage,t,ref['peaks'])
    a=np.random.default_rng(15).uniform(-1,1,(1,32,dim)).astype(np.float32)
    assert np.max(np.abs(analytic_q(a,p)-np.asarray(analytic_q_jax(jnp.asarray(a),jnp.asarray(p)))))<5e-5
   samples=target_samples(schedule('prefix',0,family),dim,32768,99)
   from .diagnostics import hist
   assert (.5*np.abs(hist(samples)-analytic_reference(schedule('prefix',0,family),dim)['target_marginals']).sum(1)).max()<.08
  if dim==1:
   x=np.linspace(-1,1,501)[:,None];p=schedule('mass',20001)
   from scipy.special import logsumexp
   expected=.25*logsumexp(np.log([.6,.2,.2])-.5*((x-np.array([-.6,0,.6]))/.1)**2-np.log(.1*np.sqrt(2*np.pi)),axis=1)
   assert np.max(abs(analytic_q(x,p)-expected))<1e-5
   assert np.max(abs(reward(x,x)-np.exp(-.5*((x-np.array([-.6,0,.6]))/.1)**2).sum(1)))<1e-12
  record['checks'].append('Target normalization, independent sampling, exact mode counts, NumPy/JAX Q;1D prior schedule/reward equivalence')
  rng=np.random.default_rng(4);x=rng.normal(size=(5,dim));y=rng.normal(size=(7,dim));w=rng.dirichlet(np.ones(7));P=exact_plan(x,y,w)
  C=((x[:,None]-y[None])**2).sum(-1);A=np.vstack([np.kron(np.eye(5),np.ones((1,7))),np.kron(np.ones((1,5)),np.eye(7))])
  result=so.linprog(C.ravel(),A_eq=A,b_eq=np.r_[np.full(5,.2),w],bounds=(0,None),method='highs')
  assert result.success and abs((P*C).sum()-result.fun)<1e-5
  record['exact_lp_cost_gap']=float(abs((P*C).sum()-result.fun));record['checks'].append('True D-dimensional exact OT agrees with independent LP')
  for method in METHODS:
   state=actor_state(method,0,dim);f=engine(method,16,64,dim,'analytic');obs=jnp.zeros((1,dim));p=jnp.asarray(schedule('prefix',0,'double'))
   tick=time.perf_counter();new,key,t,value,gn=f['step'](state,obs,jax.random.PRNGKey(1000),p);jax.block_until_ready(new.params)
   assert np.isfinite(float(value)) and np.isfinite(float(gn))
   if METHODS[method][1] is not None:
    _,ls=f['heads'](new.params,obs,t['z']);assert np.max(abs(np.asarray(ls)-np.log(METHODS[method][1])))<1e-6
   if METHODS[method][0]=='exact':
    assert np.max(abs(np.asarray(t['P']).sum(-1)-1/16))<1e-6
    assert np.max(abs(np.asarray(t['P']).sum(-2)-np.asarray(t['w'])))<1e-6
   if method=='argmax_truncated':
    reference,_,_=f['update'](state,t);assert tree_diff(new.params,reference.params)<3e-5
    assert np.all(np.abs(np.asarray(t['b']))<=1)
   elapsed=time.perf_counter()-tick
   # A warm update, separate from JIT compilation.
   tick=time.perf_counter();new,key,t,value,gn=f['step'](new,obs,key,p);jax.block_until_ready(new.params)
   record['benchmarks'].append(dict(method=method,n=16,m=64,compile_seconds=elapsed,step_seconds=time.perf_counter()-tick))
   print('validated',dim,method,flush=True);write_json(dest,record)
  import nsq.actor as module
  saved_sinkhorn,saved_exact=module.sinkhorn,module.exact_plan
  def forbidden(*a,**k):raise AssertionError('Direct GMM must not call transport')
  module.sinkhorn=module.exact_plan=forbidden
  f=engine('gmm_learned',17,68,dim,'analytic');new,*_=f['step'](actor_state('gmm_learned',0,dim),jnp.zeros((1,dim)),jax.random.PRNGKey(1),jnp.asarray(schedule('prefix',0)))
  jax.block_until_ready(new.params);module.sinkhorn,module.exact_plan=saved_sinkhorn,saved_exact
  record['checks'].append('All19 methods finite;fixed sigma constant;legacy production update equals frozen-target MSE;GMM transport forbidden')
  for n,m in SIZES[1:]:
   for method in ['gmm_learned','exact_learned','sinkhorn_e0.0001','sinkhorn_e0.1','sinkhorn_e50','argmax_truncated']:
    f=engine(method,n,m,dim,'analytic');state=actor_state(method,0,dim);key=jax.random.PRNGKey(1000);p=jnp.asarray(schedule('prefix',0));obs=jnp.zeros((1,dim))
    tick=time.perf_counter();new,key,t,value,gn=f['step'](state,obs,key,p);jax.block_until_ready(new.params);compile_sec=time.perf_counter()-tick
    assert np.isfinite(float(value)) and np.isfinite(float(gn))
    tick=time.perf_counter();new,key,t,value,gn=f['step'](new,obs,key,p);jax.block_until_ready(new.params)
    record['benchmarks'].append(dict(method=method,n=n,m=m,compile_seconds=compile_sec,step_seconds=time.perf_counter()-tick))
    print('size checked',dim,n,m,method,record['benchmarks'][-1]['step_seconds'],flush=True);write_json(dest,record)
  # Check matching-resume dynamics with actual twin TD, replay, actor and RNG.
  template=dict(stage='closed',family='tri',dim=dim,method='gmm_learned',n=16,m=64,seed=0,updates=8,parent=None,q_source=None,actor_batch=1)
  full=dict(template,name=f'validation_full_D{dim}');part=dict(template,name=f'validation_resume_D{dim}')
  run(root,full,True,max_updates=8,warmup=32,skip_evaluation=True)
  try:run(root,part,True,max_updates=8,warmup=32,interrupt_at=3,skip_evaluation=True)
  except SystemExit as e:assert e.code==75
  run(root,part,True,max_updates=8,warmup=32,skip_evaluation=True)
  a=serialization.msgpack_restore((root/'validation_runs'/full['name']/'checkpoint.msgpack').read_bytes());b=serialization.msgpack_restore((root/'validation_runs'/part['name']/'checkpoint.msgpack').read_bytes())
  for key in ['actor','critic','key','collectkey','replay','env_state','episode_step']:
   assert serialization.msgpack_serialize(a[key])==serialization.msgpack_serialize(b[key]),('resume mismatch',key)
  record['checks'].append('Complete actor/critic/Adam/replay/RNG/environment checkpoint continuation is identical')
  source=dict(template,stage='source',name=f'validation_source_D{dim}',method='sinkhorn_e0.1',actor_batch=256,updates=3)
  run(root,source,True,max_updates=3,warmup=32,skip_evaluation=True)
  src=root/'validation_runs'/source['name'];stream=QStream(src/'qstream',critic_state(0,dim).params)
  final=serialization.msgpack_restore((src/'checkpoint.msgpack').read_bytes())['critic']['params'];assert tree_diff(stream.at(3),final)==0
  replay=dict(template,stage='replay',name=f'validation_replay_D{dim}',q_source=source['name'],updates=3)
  run(root,replay,True,max_updates=3,skip_evaluation=False)
  prefix=dict(template,stage='prefix',family='double',name=f'validation_prefix_D{dim}',updates=2)
  run(root,prefix,True,max_updates=2,skip_evaluation=False)
  fork=dict(prefix,stage='mass',name=f'validation_fork_D{dim}',updates=4,parent=prefix['name'])
  run(root,fork,True,max_updates=4,skip_evaluation=True)
  record['checks'].append('Source B256;full critic parameter stream;replay diagnostics;forked analytic checkpoint')
  record.update(passed=True,completed_unix=time.time());write_json(dest,record)
 except Exception:
  record.update(error=traceback.format_exc(),failed_unix=time.time());write_json(dest,record);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dim',type=int,required=True);a=p.parse_args();main(a.root,a.dim)
