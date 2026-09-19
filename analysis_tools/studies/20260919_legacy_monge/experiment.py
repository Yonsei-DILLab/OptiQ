"""Small legacy Monge pilot; full legacy actor update and explicit assignment audit."""
import bootstrap
from pathlib import Path
import json,time,signal,argparse,os,socket
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from common.type_aliases import RLTrainState
from nsq.models import actor_state
from nsq.problems import schedule,analytic_q_jax,analytic_q,analytic_reference
from nsq.io import save,checkpoint,complete
from nsq.config import write_json
from legacy_optiq.algorithm import OptiQDIME as Original
from legacy_optiq.transport import sinkhorn
from legacy_update import LegacyMonge
from solvers import monge_plan,exact_1d_plan,hard_plan,assignment_metrics

ROOT=bootstrap.ROOT
METHODS=['argmax','exact_argmax','monge_quantile']
STOP=False

def signal_stop(*_):
 global STOP
 STOP=True

def q_apply(variables,observations,actions,**kwargs):
 q=analytic_q_jax(actions,variables['params'])
 return jnp.stack((q,q))[...,None]

PARAMS=dict(num_policy_samples=256,proposals_per_policy_sample=4,proposal_sampling_mode='stratified',proposal_std=.2,
 proposal_clip=.5,include_anchor=False,density_correction=True,density_beta=1.,adaptive_density_beta=False,
 minimum_source_ess=256.,density_beta_grid_size=257,temperature=.25,sinkhorn_epsilon=.05,sinkhorn_iterations=30,
 source_q_eval='mean',return_clouds=True)

def make_update(method,original=False,n=256,m=1024):
 params=dict(PARAMS,num_policy_samples=n,proposals_per_policy_sample=m//n,minimum_source_ess=float(n))
 algorithm=Original if original else LegacyMonge
 @jax.jit
 def update(state,key,p):
  oracle=RLTrainState.create(apply_fn=q_apply,params=p,target_params=p,batch_stats={},target_batch_stats={},tx=optax.identity())
  new,loss,key,cloud=algorithm.update_actor(state,oracle,jnp.zeros((1,1)),key,jnp.ones(1),transport_target_mode=method,**params)
  x=cloud['cloud_transport_sources'][0];b=cloud['cloud_proposals'][0];w=cloud['cloud_weights'][0];P=cloud['cloud_transport'][0]
  met=assignment_metrics(x,b,w,P)
  met.update(loss=loss,ess=cloud['source_ess_absolute'],wmax=cloud['max_source_weight'],
    actor_shift=cloud['selected_delta_l2'],source_spread=cloud['policy_spread_l2'])
  return new,key,cloud,met
 return update

@jax.jit
def draw(state,z):
 raw=state.apply_fn({'params':state.params},jnp.zeros_like(z),z)
 return jnp.clip(raw,-1,1),raw


def evaluate(out,task,step,state,cloud,p):
 # Independent 32768 action draws at each saved evaluation, identical keys across methods.
 z=jax.random.normal(jax.random.PRNGKey(1100000+task['seed']*100000+step),(32768,1))
 actions,raw=jax.device_get(draw(state,z));ref=analytic_reference(p,1);edges=ref['edges'];target=ref['target_marginals'][0]
 mass=np.histogram(actions[:,0],edges)[0]/len(actions);bound=ref['boundaries'];bm=np.histogram(actions[:,0],bound)[0]/len(actions)
 selected=np.asarray(cloud['cloud_selected'])[0,:,0];b=np.asarray(cloud['cloud_proposals'])[0,:,0];w=np.asarray(cloud['cloud_weights'])[0]
 wm=np.histogram(b,edges,weights=w)[0];pm=np.histogram(b,edges)[0]/len(b);sm=np.histogram(selected,edges)[0]/len(selected)
 tbm=np.histogram(b,bound,weights=w)[0];sbm=np.histogram(selected,bound)[0]/len(selected)
 metrics=dict(step=step,hist_tv=.5*np.abs(mass-target).sum(),basin_tv=.5*np.abs(bm-ref['target_basin_mass']).sum(),
  teacher_hist_tv=.5*np.abs(wm-target).sum(),teacher_basin_tv=.5*np.abs(tbm-ref['target_basin_mass']).sum(),
  selected_teacher_basin_tv=.5*np.abs(sbm-tbm).sum(),actor_selected_basin_tv=.5*np.abs(bm-sbm).sum(),
  backup_bias=float(analytic_q(actions,p).mean()-ref['target_backup']),boundary_fraction=float(np.mean(np.abs(raw)>=1)))
 save(out/'density'/f'{step:06d}.npz',dict(**metrics,actions=actions,edges=edges,target=target,actor=mass,
  proposal=pm,teacher=wm,selected=sm,basin=bm,target_basin=ref['target_basin_mass'],teacher_basin=tbm,selected_basin=sbm,
  boundaries=bound,q_grid=ref['grid'],q=ref['q_slice'],b=b,w=w,selected_actions=selected))
 return metrics


def audit(out,step,cloud):
 """All assignment alternatives on ONE identical source/candidate/weight cloud."""
 x=np.asarray(cloud['cloud_transport_sources'])[0];b=np.asarray(cloud['cloud_proposals'])[0];w=np.asarray(cloud['cloud_weights'])[0]
 xj,bj,wj=map(jnp.asarray,(x,b,w));cost=jnp.square(xj[:,None]-bj[None]).sum(-1);cost=cost/(cost.mean()+1e-8)
 P=sinkhorn(cost[None],wj[None],.05,30)[0];E=exact_1d_plan(xj,bj,wj);M=monge_plan(xj,bj,wj)
 # Same quantized teacher: isolates the assignment heuristic from target resampling.
 quantw=M.sum(0);S=sinkhorn(cost[None],quantw[None],.05,30)[0]
 data=dict(x=x,b=b,w=w,log_density=np.asarray(cloud['cloud_log_density'])[0],quantized_w=np.asarray(quantw))
 order=np.argsort(x[:,0],kind='stable');indices=np.asarray(M).argmax(1)[order]
 bijection=np.zeros((len(x),len(x)),np.float32);bijection[order,np.arange(len(x))]=1/len(x)
 data.update(monge_bijection=bijection,monge_representatives=b[indices],monge_representative_candidate_index=indices)
 for name,plan in [('sinkhorn',P),('exact',E),('monge',M),('sinkhorn_same_quantized_teacher',S)]:
  data[name+'_plan']=np.asarray(plan);data[name+'_effective']=np.asarray(hard_plan(plan))
  data.update({name+'_'+k:np.asarray(v) for k,v in assignment_metrics(xj,bj,wj,plan).items()})
 save(out/'assignment_audits'/f'{step:06d}.npz',data)


def run(task,out=None,limit=None,interrupt_at=None,diagnostics=True):
 global STOP
 STOP=False;code=bootstrap.verify();out=Path(out or ROOT/'runs'/task['name']);out.mkdir(parents=True,exist_ok=True)
 steps=limit or task['updates'];pconf=json.loads((ROOT/'DEPLOYMENT.json').read_text())
 for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,signal_stop)
 if (out/'COMPLETE.json').exists():
  assert json.loads((out/'COMPLETE.json').read_text())['source_code_id']==code
  return
 settings={**PARAMS,**task,'num_policy_samples':task['n'],'proposals_per_policy_sample':task['m']//task['n'],'minimum_source_ess':float(task['n'])}
 write_json(out/'config.json',dict(**settings,source_code_id=code,commit=pconf['commit'],base_source=bootstrap.SPEC,
    proposal_count=task['m'],monge_representatives=task['n'] if task['method']=='monge_quantile' else None,
    monge_problem='Equal-weight indexed latent particles -> quantized target; original weighted teacher not preserved exactly'))
 write_json(out/'RUNNING.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
   array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),started=time.time(),commit=pconf['commit'],source_code_id=code))
 state=actor_state('argmax_truncated',task['seed'],1);key=jax.random.PRNGKey(1000+task['seed']);start=0;elapsed=0.;train=0.;metrics=[];evs=[]
 cp=out/'checkpoint.msgpack'
 if cp.exists():
  ck=serialization.msgpack_restore(cp.read_bytes());assert ck['source_code_id']==code
  state=serialization.from_state_dict(state,ck['actor']);key=jnp.asarray(ck['key']);start=int(ck['step']);elapsed=float(ck['elapsed']);train=float(ck['train_seconds'])
 update=make_update(task['method'],n=task['n'],m=task['m']);family,stage=task['case'].split('_');begin=time.time();first=start+1
 def flush(step):
  nonlocal metrics,evs,first
  if metrics:save(out/'metrics'/f'{first:06d}.npz',{k:np.asarray([d[k] for d in metrics]) for k in metrics[0]})
  if evs:save(out/'evaluations'/f'{first:06d}.npz',{k:np.asarray([d[k] for d in evs]) for k in evs[0]})
  ck=dict(source_code_id=code,actor=serialization.to_state_dict(state),key=np.asarray(key),step=step,elapsed=elapsed+time.time()-begin,train_seconds=train)
  checkpoint(cp,ck);write_json(out/'progress.json',dict(name=task['name'],step=step,total=steps,elapsed=ck['elapsed'],train_seconds=train,source_code_id=code,commit=pconf['commit']))
  metrics=[];evs=[];first=step+1
 for step in range(start+1,steps+1):
  p=jnp.asarray(schedule(stage,step,family));tick=time.perf_counter();state,key,cloud,met=update(state,key,p);jax.block_until_ready(state.params);train+=time.perf_counter()-tick
  met={k:float(v) for k,v in jax.device_get(met).items()};assert all(np.isfinite(v) for v in met.values()),met
  if task['method']=='monge_quantile':assert met['selected_teacher_cdf_error']<=.5/task['n']+1e-5
  metrics.append(dict(step=step,train_seconds=train,**met))
  near=any(abs(step-t)<=100 for t in [20000,22000,25000,27000,30000])
  if diagnostics and (step==1 or step%500==0 or (near and step%20==0) or step==steps):
   evs.append(evaluate(out,task,step,state,cloud,np.asarray(p)))
  if diagnostics and step in [1,20000,20001,22000,25001,27000,30001,35000]:audit(out,step,cloud)
  if step==interrupt_at:STOP=True
  if step%200==0 or step==steps or STOP:flush(step)
  if step%5000==0 or step==steps:print(task['name'],step,flush=True)
  if STOP:raise SystemExit(75)
 (out/'final_actor.msgpack').write_bytes(serialization.to_bytes(state.params))
 complete(out,dict(task=task,step=steps,source_code_id=code,commit=pconf['commit'],base_commit=bootstrap.SPEC['commit'],completed_unix=time.time()))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--index',type=int,required=True);args=a.parse_args();assert jax.default_backend()=='gpu'
 task=json.loads((ROOT/'tasks.json').read_text())[args.index]
 try:run(task)
 except Exception:
  import traceback
  write_json(ROOT/'runs'/task['name']/'FAILED.json',dict(error=traceback.format_exc(),time=time.time()));raise
