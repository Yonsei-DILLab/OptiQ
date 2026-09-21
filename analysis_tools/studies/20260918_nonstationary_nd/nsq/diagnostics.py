"""Direct actor histograms, independent references, and tracking-speed diagnostics."""
import functools,time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
import jax
import jax.numpy as jnp
from .models import actor_apply,q_reference_mean,sample_action
from .problems import GRID,EDGES,analytic_reference,target_samples,analytic_q,ORTHO_STD
from .config import SAMPLES,raw_step,METHODS
from .io import save

@functools.lru_cache(None)
def sampler(method,dim):
 apply=actor_apply(method,dim);legacy=METHODS[method][0]=='legacy'
 @jax.jit
 def sample(params,obs,key):
  zk,ek=jax.random.split(key);z=jax.random.normal(zk,(SAMPLES,dim));eps=jax.random.normal(ek,(SAMPLES,dim))
  out=apply({'params':params},jnp.broadcast_to(obs,(SAMPLES,dim)),z)
  mu,ls=(out,jnp.full_like(out,-np.inf)) if legacy else out
  return dict(actions=jnp.clip(mu,-1,1) if legacy else jnp.tanh(mu+jnp.exp(ls)*eps),mu=mu,log_sigma=ls)
 return sample

def hist(a,weights=None,edges=EDGES):
 return np.stack([np.histogram(a[:,d],edges,weights=weights)[0] for d in range(a.shape[1])])/(len(a) if weights is None else np.sum(weights))

@functools.lru_cache(None)
def sobol(dim):return (2*qmc.Sobol(dim,scramble=True,seed=938177).random_base2(17)-1).astype(np.float32)

def qvalues(params,obs,actions):
 return np.concatenate([np.asarray(q_reference_mean(params,jnp.asarray(obs)[None],jnp.asarray(a)[None]))[0] for a in np.array_split(actions,max(1,int(np.ceil(len(actions)/8192))))])

def learned_reference(params,dim,seed):
 obs=np.zeros(dim,np.float32)
 if dim==1:
  grid=GRID[:,None].astype(np.float32);q=qvalues(params,obs,grid).astype(float);exp=np.exp((q-q.max())/.25)
  areas=(exp[:-1]+exp[1:])/2*np.diff(GRID);cdf=np.r_[0,np.cumsum(areas)];cdf/=cdf[-1]
  mass=np.diff(np.interp(EDGES,GRID,cdf))[None];pdf=exp/np.trapz(exp,GRID)
  from scipy.signal import find_peaks
  peaks=find_peaks(pdf)[0]
  bound=np.unique(np.r_[-1,[GRID[i+np.argmin(pdf[i:j+1])] for i,j in zip(peaks[:-1],peaks[1:])],1])
  samples=np.interp(np.random.default_rng(670001+seed).random(SAMPLES),cdf,GRID)[:,None].astype(np.float32)
  return dict(grid=GRID,q_slice=q,pdf_first=pdf,edges=EDGES,target_marginals=mass,boundaries=bound,
   target_basin_mass=np.diff(np.interp(bound,GRID,cdf)),peaks=GRID[peaks],target_backup=np.trapz(pdf*q,GRID),
   reference_kind='learned_Q_8193_grid',reference_ess=0.,reference_reliable=True,reference_half_tv=0.),samples,None,None
 points=sobol(dim);q=qvalues(params,obs,points).astype(float);w=np.exp(q/.25-logsumexp(q/.25));marg=hist(points,w)
 ess=float(1/np.square(w).sum());half=len(points)//2
 m1=hist(points[:half],w[:half]);m2=hist(points[half:],w[half:]);tv=float(.5*np.abs(m1-m2).sum(-1).max())
 ref_rng=np.random.default_rng(670001+seed);ix=ref_rng.choice(len(points),SAMPLES,p=w);samples=points[ix]
 # The first-coordinate marginal from independent importance quadrature, not a joint density.
 pdf=marg[0]/np.diff(EDGES);from scipy.signal import find_peaks
 peaks=find_peaks(pdf,prominence=max(pdf.max()*.03,1e-12))[0]
 centers=(EDGES[1:]+EDGES[:-1])/2
 bound=np.unique(np.r_[-1,[centers[i+np.argmin(pdf[i:j+1])] for i,j in zip(peaks[:-1],peaks[1:])],1])
 basin=np.histogram(points[:,0],bound,weights=w)[0];slicepoints=np.zeros((len(GRID),dim),np.float32);slicepoints[:,0]=GRID
 ref=dict(grid=GRID,q_slice=qvalues(params,obs,slicepoints),pdf_first=pdf,edges=EDGES,target_marginals=marg,
  boundaries=bound,target_basin_mass=basin,peaks=centers[peaks],target_backup=float(w@q),reference_kind='learned_Q_Sobol_uniform_IS_131072',
  reference_ess=ess,reference_reliable=bool(ess>=512 and tv<=.05),reference_half_tv=tv)
 return ref,samples,points,w

def sliced_w1(a,b,seed):
 rng=np.random.default_rng(seed);v=rng.normal(size=(a.shape[1],32));v/=np.linalg.norm(v,axis=0)
 # Same fixed subsample and projections across methods and checkpoints.
 return float(np.mean(np.abs(np.sort(a[:4096]@v,axis=0)-np.sort(b[:4096]@v,axis=0))))

def evaluate(out,task,step,state,f,qarg,qkind,final=False):
 dim=task['dim'];obs=np.zeros(dim,np.float32);seed=task['seed'];legacy=METHODS[task['method']][0]=='legacy'
 if qkind=='analytic':
  ref=analytic_reference(qarg,dim);targets=target_samples(qarg,dim,SAMPLES,670001+seed);points=weights=None
 else:ref,targets,points,weights=learned_reference(qarg,dim,seed)
 evalkey=jax.random.PRNGKey((970001 if final else 710001)+seed)
 d=jax.device_get(sampler(task['method'],dim)(state.params,jnp.asarray(obs),evalkey));a=d['actions'];mass=hist(a)
 bm=np.histogram(a[:,0],ref['boundaries'])[0]/SAMPLES
 backup=float(np.asarray(f['qvalue'](jnp.asarray(obs)[None],jnp.asarray(a)[None],qarg)).mean())
 mtv=.5*np.abs(mass-ref['target_marginals']).sum(-1)
 metrics=dict(step=step,first_axis_tv=float(mtv[0]),mean_marginal_tv=float(mtv.mean()),max_marginal_tv=float(mtv.max()),
  basin_mass_tv=float(.5*np.abs(bm-ref['target_basin_mass']).sum()),sliced_w1=sliced_w1(a,targets,89011),
  backup=backup,target_backup=float(ref['target_backup']),backup_bias=backup-ref['target_backup'],
  sigma_mean=0. if legacy else float(np.exp(d['log_sigma']).mean()),between_mu_variance=float(np.var(d['mu'],axis=0).mean()),
  within_variance=0. if legacy else float(np.exp(2*d['log_sigma']).mean()),target_peaks=len(ref['peaks']),
  reference_ess=ref['reference_ess'],reference_reliable=ref['reference_reliable'],reference_half_tv=ref.get('reference_half_tv',0.))
 if dim>1:metrics['transverse_tv']=float(mtv[1:].mean())
 t,_=f['teacher'](state.params,jnp.asarray(obs)[None],jax.random.PRNGKey(850001+seed+step),qarg)
 b,w=np.asarray(t['b'])[0],np.asarray(t['w'])[0];pm=hist(b);wm=hist(b,w);tbm=np.histogram(b[:,0],ref['boundaries'],weights=w)[0]
 assign=f['assignment'](state.params,t)
 metrics.update(teacher_first_tv=float(.5*np.abs(wm[0]-ref['target_marginals'][0]).sum()),teacher_marginal_tv=float(.5*np.abs(wm-ref['target_marginals']).sum(-1).mean()),
  teacher_basin_mass_tv=float(.5*np.abs(tbm-ref['target_basin_mass']).sum()),actor_teacher_tv=float(.5*np.abs(mass-wm).sum(-1).mean()),
  common_marginal_nll=float(assign['common_nll']),usage_ess=float(assign['usage_ess']),underused_fraction=float(assign['underused_fraction']),row_entropy=float(assign['row_entropy']),probe_ess=float(1/np.square(w).sum()),probe_wmax=float(w.max()))
 # Small files every evaluation; full action/parameter samples only at registered snapshots.
 saved=dict(**ref,marginal_mass=mass,basin_mass=bm,proposal_marginals=pm,teacher_marginals=wm,teacher_basin_mass=tbm,
  step=np.array(step),evaluation_key=np.asarray(evalkey),sample_count=SAMPLES,fixed_mu=d['mu'][:128],fixed_log_sigma=d['log_sigma'][:128],
  **{k:np.asarray(v) for k,v in metrics.items() if k not in ('step','target_backup','reference_ess','reference_reliable','reference_half_tv')})
 if dim==2:
  edges2=np.linspace(-1,1,65);joint=np.histogram2d(a[:,0],a[:,1],bins=[edges2,edges2])[0]/SAMPLES
  if qkind=='analytic':
   from .problems import cdf,transverse_cdf
   jt=np.outer(np.diff(cdf(edges2,qarg)),np.diff(transverse_cdf(edges2)))
  else:jt=np.histogram2d(points[:,0],points[:,1],bins=[edges2,edges2],weights=weights)[0]
  saved.update(joint_edges=edges2,joint_mass=joint,target_joint_mass=jt)
  metrics['joint_2d_tv']=float(.5*np.abs(joint-jt).sum());saved['joint_2d_tv']=metrics['joint_2d_tv']
 if raw_step(step) or final:saved.update(actions=a,target_samples=targets,mu=d['mu'],log_sigma=d['log_sigma'],proposal_actions=b,teacher_weights=w)
 suffix='final_independent' if final else f'{step:06d}'
 save(out/'density'/f'{suffix}.npz',saved)
 if step in {20000,20001,25001,30001,35000} or final:
  raw={k:np.asarray(v) for k,v in t.items() if k not in ['student_eps','component']}
  raw.update({k:np.asarray(v) for k,v in assign.items() if k!='A'})
  raw.update(step=step,source_order=np.argsort(np.asarray(t['positions'])[0,:,0],kind='stable'),candidate_order=np.argsort(b[:,0],kind='stable'))
  save(out/'assignments'/f'{suffix}.npz',raw)
 return metrics

@functools.lru_cache(None)
def return_engine(dim):
 @jax.jit
 def run(actor,key):
  def step(carry,_):
   obs,key,total=carry;key,ak=jax.random.split(key);a=sample_action(actor,obs,ak,deterministic=False)
   g=jnp.exp(-.5*((a[...,0,None]-jnp.array([-.6,0,.6]))/.1)**2-.5*jnp.square(a[...,1:]/ORTHO_STD).sum(-1)[...,None]).sum(-1)
   return (a,key,total+g-.25*jnp.square(a-obs).mean(-1)),None
  (_,_,returns),_=jax.lax.scan(step,(jnp.zeros((10,dim)),key,jnp.zeros(10)),None,length=200)
  return returns
 return run

def policy_returns(state,seed,dim):
 r=np.asarray(return_engine(dim)(state,jax.random.PRNGKey(112300+seed)))
 return dict(stochastic_return_mean=float(r.mean()),stochastic_return_sd=float(r.std(ddof=1)),returns=r)
