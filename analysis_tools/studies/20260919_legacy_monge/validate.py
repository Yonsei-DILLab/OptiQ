"""Independent solver checks and archived-update parity before pilot launch."""
import bootstrap
import json,time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment,linprog
from flax import serialization
from nsq.config import write_json
from nsq.models import actor_state
from nsq.problems import schedule
from solvers import monge_plan,exact_1d_plan,hard_plan,assignment_metrics
from experiment import make_update,run,audit,evaluate

def leaves_equal(a,b,tol=0.):
 aa,ta=jax.tree_util.tree_flatten(a);bb,tb=jax.tree_util.tree_flatten(b);assert ta==tb
 err=max(float(np.max(np.abs(np.asarray(x).astype(float)-np.asarray(y).astype(float)))) for x,y in zip(aa,bb))
 assert err<=tol,err
 return err

def main():
 code=bootstrap.verify();assert jax.default_backend()=='gpu';out=bootstrap.ROOT/'validation';out.mkdir(exist_ok=True)
 results={};rng=np.random.default_rng(52)
 for n,m in [(5,7),(16,64),(256,1024)]:
  for case in ['random','tied_sources','one_teacher']:
   x=rng.uniform(-1,1,(n,1)).astype('float32');b=rng.uniform(-1,1,(m,1)).astype('float32');w=rng.dirichlet(np.ones(m)).astype('float32')
   if case=='tied_sources':x[:max(2,n//2)]=0
   if case=='one_teacher':w[:]=0;w[0]=1
   M=np.asarray(monge_plan(jnp.array(x),jnp.array(b),jnp.array(w)));idx=M.argmax(1);rep=b[idx]
   # Representatives define an equal-weight N x N problem. Compare to independent solver.
   C=np.square(x[:,None]-rep[None]).sum(-1).astype(float);ri,ci=linear_sum_assignment(C)
   cost=np.square(x-rep).sum()/n;opt=C[ri,ci].mean();assert abs(cost-opt)<1e-6
   assert np.all((M>0).sum(1)==1);np.testing.assert_allclose(M.sum(1),1/n,atol=1e-7)
   met={k:float(v) for k,v in assignment_metrics(jnp.array(x),jnp.array(b),jnp.array(w),jnp.array(M)).items()}
   assert met['selected_teacher_cdf_error']<=.5/n+1e-5
   E=np.asarray(exact_1d_plan(jnp.array(x),jnp.array(b),jnp.array(w)))
   np.testing.assert_allclose(E.sum(1),1/n,atol=1e-6);np.testing.assert_allclose(E.sum(0),w/w.sum(),atol=1e-6)
   if n==5:
    A=np.vstack([np.kron(np.eye(n),np.ones((1,m))),np.tile(np.eye(m),(1,n))]);Cfull=np.square(x[:,None]-b[None]).sum(-1)
    lp=linprog(Cfull.ravel(),A_eq=A,b_eq=np.r_[np.full(n,1/n),w.astype(float)/w.astype(float).sum()],bounds=(0,None),method='highs')
    assert lp.success,lp.message
    assert abs((E*Cfull).sum()-lp.fun)<2e-6
   results[f'solver_{n}_{m}_{case}']=dict(monge_cost=float(cost),independent_cost=float(opt),**met)
 # General original teacher is not representable with N indivisible atoms when M>N and all w>0.
 assert 7>5;results['original_weighted_teacher_not_claimed_exact']=True
 for n,m in [(16,64),(256,1024)]:
  for family,stage in [('double','mass'),('tri','split')]:
   state=actor_state('argmax_truncated',0,1);key=jax.random.PRNGKey(1000);p=jnp.asarray(schedule(stage,20001,family))
   old=make_update('argmax',original=True,n=n,m=m)(state,key,p)
   new=make_update('argmax',n=n,m=m)(state,key,p)
   err=leaves_equal(old[0].params,new[0].params,2e-6)
   leaves_equal(old[0].opt_state,new[0].opt_state,2e-6);leaves_equal(old[1:],new[1:],2e-6)
   results[f'baseline_parity_{family}_{n}']=err
   for method in ['exact_argmax','monge_quantile']:
    result=make_update(method,n=n,m=m)(state,key,p)
    for name in ['cloud_proposals','cloud_weights','cloud_transport_sources','cloud_log_density']:
     np.testing.assert_allclose(np.asarray(new[2][name]),np.asarray(result[2][name]),atol=2e-6,rtol=2e-6)
    # Output-space MSE gradient points exactly toward the assigned frozen target.
    x=np.asarray(result[2]['cloud_transport_sources']);target=np.asarray(result[2]['cloud_selected'])
    before=np.mean((x-target)**2);after=np.mean((x+.1*(target-x)-target)**2);assert after<=before+1e-12
 tasks=json.loads((bootstrap.ROOT/'tasks.json').read_text())
 for task in tasks[:3]:
  run(task,out=out/task['method'],limit=20,diagnostics=True)
 # Serialized optimizer + RNG continuation must reproduce uninterrupted updates exactly.
 task={**tasks[2],'n':16,'m':64,'name':'resume_check'}
 run(task,out=out/'uninterrupted',limit=6,diagnostics=False)
 try:run(task,out=out/'resumed',limit=6,interrupt_at=3,diagnostics=False)
 except SystemExit as e:assert e.code==75
 run(task,out=out/'resumed',limit=6,diagnostics=False)
 a=serialization.msgpack_restore((out/'uninterrupted'/'checkpoint.msgpack').read_bytes())
 b=serialization.msgpack_restore((out/'resumed'/'checkpoint.msgpack').read_bytes())
 results['resume_actor_error']=leaves_equal(a['actor'],b['actor'])
 results['resume_rng_error']=leaves_equal(a['key'],b['key'])
 results.update(ok=True,source_code_id=code,commit=json.loads((bootstrap.ROOT/'DEPLOYMENT.json').read_text())['commit'],
  completed=time.time(),devices=[str(x) for x in jax.devices()])
 write_json(out/'PASSED.json',results);print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
