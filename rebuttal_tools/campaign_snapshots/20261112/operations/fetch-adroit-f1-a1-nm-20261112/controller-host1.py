"""Capacity-bounded F1/A1 runner. No restart, seed replacement or performance-based retry."""
from pathlib import Path
from collections import defaultdict
import os,json,subprocess,time,hashlib,fcntl,signal,shutil,traceback
ROOT=Path(__file__).resolve().parent

def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default

def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

def identity(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return int(a[19]) if a[0]!='Z' else None
 except FileNotFoundError:return None

def cores(active):
 groups={}
 for c in sorted(os.sched_getaffinity(0)):
  p=Path(f'/sys/devices/system/cpu/cpu{c}/topology');k=((p/'physical_package_id').read_text(),(p/'core_id').read_text());groups.setdefault(k,[]).append(c)
 busy=set(sum(list(groups.values())[:4],[]))
 for j in active:
  busy.update(j['learner_cpus']);busy.update(sum(j['worker_cpus'],[]))
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    a=os.sched_getaffinity(int(p.name))
    if len(a)<=16:busy.update(a)
   except (PermissionError,ProcessLookupError):pass
 return [g for g in groups.values() if not busy.intersection(g)]

def gpu_state():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 occupied=defaultdict(set)
 for l in procs.splitlines():
  pid,uuid=map(str.strip,l.split(','));occupied[uuid].add(int(pid))
 return [dict(index=int(a),uuid=b.strip(),free=int(c),util=int(d),pids=occupied[b.strip()]) for a,b,c,d in (l.split(',') for l in raw.splitlines())]

def verify(j):
 o=Path(j['output']);c=read(o/'config.json')
 if c is None:return
 assert c['method']==j['method'] and c['seed']==j['seed'] and c['batch_size']==256 and c['num_envs']==256
 assert c['utd']==(.125 if j['family']=='fetch' else .25)
 if j['family']=='adroit':
  assert c['actor_hidden']==c['critic_hidden']==[256,256] and c['critic_layer_norm']
  assert c['demonstration_dataset_sha256']==j['demo_sha256'] and c['offline_pretraining_updates']==0
  proof=read(o/'demonstration-validation.json');assert proof['passed'] and proof['replayed'] and proof['successful_trajectories']==25
 else:
  assert c['normalization_recipe'].startswith('Completed-episode HER')
  if j.get('protocol_variant')=='F1-slide-20cm':
   assert c.get('distance_threshold_m')==.20 and c.get('horizon')==50
   assert c.get('protocol_variant')==j['protocol_variant']
 if j['method']=='ibolt':
  cfg=read(o/('mujoco-config.json' if j['family']=='fetch' else 'ibolt-config.json'))
  if cfg:
   if j['phase'].startswith('entropy_'):
    assert cfg['dacer']['target_entropy_per_dim']==j['dacer_target_entropy_per_dim']
    assert cfg['dacer']['noise_scale']==(2. if j['family']=='fetch' else .1)
    assert cfg['dacer']['alpha_lr']==.03 and cfg['dacer']['initial_alpha']==.27 and cfg['dacer']['interval_updates']==10000
   if j['phase']=='nm_pilot':
    assert cfg['dacer']['target_entropy_per_dim']==-.9
    assert cfg['alg']['actor']['num_policy_samples']==j['num_policy_samples']
    assert cfg['alg']['actor']['proposals_per_policy_sample']==4
    assert cfg['alg']['actor']['num_policy_samples']*4==j['num_reference_actions']
    assert c['num_reference_actions']==j['num_reference_actions']
   assert cfg['dacer']['enabled'] and cfg['temperature']==cfg['alg']['actor']['temperature']==j['temperature']
   assert cfg['alg']['actor']['hidden_dims']==cfg['alg']['critic']['hs']==[256,256]
   assert bool(cfg['alg']['critic'].get('layer_norm',False))==(j['family']=='adroit')
 if j['method']=='sac' and j['family']=='adroit':assert c['learner']=='JAX SAC' and c['backend']=='gpu'
 if j['method']=='sac' and j['family']=='fetch':
  a=read(o/'architecture-audit.json')
  if a:assert a['backend']=='gpu' and a['actor_hidden_dims']==a['critic_hidden_dims']==[256,256] and not a['layer_norm']
 j['config_verified']=True

def placement(lanes,p):
 if p.get('physical_cores_per_job',6)==3:
  # Four unchanged collector processes share the four SMT threads of two cores.
  worker_threads=lanes[1]+lanes[2]
  assert len(worker_threads)>=4, 'Compact placement requires SMT pairs'
  return lanes[0],[[c] for c in worker_threads[:4]]
 return lanes[0]+lanes[1],lanes[2:6]

def launch(j,p,g,lanes):
 learner,workers=placement(lanes,p)
 j.update(gpu=g['index'],gpu_uuid=g['uuid'],learner_cpus=learner,worker_cpus=workers,started_unix=time.time())
 spec=ROOT/'specs'/(j['id']+'.json');write(spec,j)
 source=Path(j.get('source',ROOT/(j['family']+'-source')))
 if j['phase'].startswith('entropy_') or j['phase']=='nm_pilot':assert hashlib.sha256(Path(j['source_manifest']).read_bytes()).hexdigest()==j['source_manifest_sha256']
 env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=str(g['index']),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled',JAX_COMPILATION_CACHE_DIR=str(ROOT/'jax-cache'),MPLCONFIGDIR=str(ROOT/'mpl-cache'))
 module='adroit_rebuttal.run' if j['family']=='adroit' else 'rebuttal_tools.fetch_jax_sac_worker' if j['method']=='sac' else 'rebuttal_tools.fetch_suite_5m_worker'
 args=['nice','-n','10','taskset','-c',','.join(map(str,j['learner_cpus'])),p['python'][j['family']],'-u','-m',module,'--spec',str(spec)]
 if j['phase']=='smoke':args.append('--smoke')
 with (ROOT/'logs'/(j['id']+'.log')).open('x') as log:
  proc=subprocess.Popen(args,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 j.update(pid=proc.pid,start_ticks=identity(proc.pid),status='running');return proc

def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(ROOT/'plan.json');jobs=p['jobs'];children={}
 assert read(ROOT/'preflight.json')['passed']
 for fam,digest in p['source_manifests'].items():assert hashlib.sha256((ROOT/(fam+'-source')/'source-sha256.json').read_bytes()).hexdigest()==digest
 def stop(sig,frame):
  for j in jobs:
   if j['status']=='running' and identity(j['pid'])==j['start_ticks']:os.killpg(j['pid'],signal.SIGTERM);j['status']='stopped_by_user'
  p['status']='stopped_by_user';write(ROOT/'plan.json',p);raise SystemExit
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 while True:
  for j in jobs:
   if j['status'] in ('waiting_temperature','waiting_entropy'):
    from entropy_dependencies import resolve_dependency
    resolve_dependency(j,ROOT,read)
   if j['status']=='waiting_selection':
    selected=read(ROOT/'selections'/(j['family']+'-'+j['task']+'.json'))
    if selected:j['status']='queued' if j['temperature']==selected['temperature'] else 'not_selected'
   if j['status']!='running':continue
   verify(j);proc=children.get(j['id'])
   live=(proc.poll() is None) if proc else identity(j['pid'])==j['start_ticks']
   prog=read(Path(j['output'])/'progress.json',{});j['progress']=prog
   if live:continue
   done=read(Path(j['output'])/('completed.json' if j['family']=='adroit' else 'verified-completion.json'))
   expected=(13568 if j['family']=='adroit' else 13312) if j['phase']=='smoke' else 5000192
   if done and done.get('completed') and done['steps']==expected:
    assert done['updates']==int((expected-12800)*(.125 if j['family']=='fetch' else .25))
    j.update(status='completed',result=done)
   else:j.update(status='failed',failure=read(Path(j['output'])/'failure.json'),exit_code=None if proc is None else proc.returncode)
   print(j['id'],j['status'],flush=True)
  active=[j for j in jobs if j['status']=='running'];own_by_gpu=defaultdict(set)
  for j in active:own_by_gpu[j['gpu_uuid']].add(j['pid'])
  # A failure never creates a new seed/retry. It blocks that task/method until reviewed.
  failed={(j['family'],j['task'],j['method']) for j in jobs if j['status']=='failed'}
  queued=[j for j in jobs if j['status']=='queued' and (j['family'],j['task'],j['method']) not in failed]
  def gate(j):
   if j['phase']=='nm_pilot':
    proof=read(ROOT/'nm-archive-ready.json',{})
    if not proof.get('passed') or proof.get('source_manifests',{}).get(Path(j['source']).name)!=j['source_manifest_sha256']:return False
   if j.get('protocol_variant')=='F1-slide-20cm':
    proof=read(ROOT/'slide-20cm-20261112/environment-verification.json',{})
    if not proof.get('passed') or proof.get('distance_threshold_m')!=.20:return False
   if j['phase'].startswith('entropy_'):
    proof=read(ROOT/'entropy-archive-ready.json',{})
    if not proof.get('passed') or proof.get('source_manifests',{}).get(Path(j['source']).name)!=j['source_manifest_sha256']:return False
   if j['phase']=='smoke':return True
   if j['family']=='adroit':return True  # Unmodified, already numerically and environmentally verified source.
   result=read(ROOT/'fetch-smoke-gates.json',{})
   return result.get(j['task']+'-'+j['method'],False)
  queued=[j for j in queued if gate(j)]
  free=shutil.disk_usage(ROOT).free
  space=free>max(p['min_free_gib']*2**30,(len(active)+1)*p.get('disk_reserve_mib_per_job',512)*2**20+2*2**30)
  archive=read(ROOT/'archive-ready.json')
  if queued and space and archive and archive.get('passed') and not (ROOT/'PAUSE').exists():
   admission=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX)
   try:
    lanes=cores(active);need=p.get('physical_cores_per_job',6)
    # Spread admissions over less occupied GPUs before adding another co-resident learner.
    available_gpus=sorted(gpu_state(),key=lambda g:(len(own_by_gpu[g["uuid"]]),g["util"],-g["free"],g["index"]))
    for g in available_gpus:
     if g['index'] in p.get('excluded_gpus',[]):continue
     own=own_by_gpu[g['uuid']]
     # Other users' GPU allocations are entirely excluded, even when utilization is low.
     foreign=g['pids']-own
     if foreign or len(own)>=p['max_jobs_per_gpu'] or len(lanes)<need or g['free']<10000:continue
     if own and g['util']>80 and p.get('wait_for_low_gpu_util',True):continue
     if not queued:break
     j=queued.pop(0);children[j['id']]=launch(j,p,g,lanes[:need]);lanes=lanes[need:]
     print('START',j['id'],'GPU',g['index'],flush=True)
   finally:fcntl.flock(admission,fcntl.LOCK_UN);admission.close()
  p['counts']={s:sum(j['status']==s for j in jobs) for s in set(j['status'] for j in jobs)};p['updated_unix']=time.time();p['free_gib']=free/2**30;p['status']='running';write(ROOT/'plan.json',p)
  time.sleep(15)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not isinstance(e,(SystemExit,KeyboardInterrupt)):write(ROOT/'controller-failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
  raise
