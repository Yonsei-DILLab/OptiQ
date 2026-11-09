"""Durable dependency- and GPU-gated dispatch; training jobs are managed by Supervisor."""
import argparse,csv,fcntl,json,os,subprocess,time
from pathlib import Path

ACTIVE={'RUNNING','STARTING','STOPPING','BACKOFF'}

def atomic_json(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def supervisor_states():
 r=subprocess.run(['sudo','-n','supervisorctl','status'],capture_output=True,text=True,timeout=20)
 if r.returncode not in (0,3):raise RuntimeError('Supervisor status unavailable: '+r.stderr+r.stdout)
 result={}
 for line in r.stdout.splitlines():
  fields=line.split()
  if len(fields)>=2:result[fields[0]]=fields[1]
 return result

def prerequisites_ready(config,states):
 for prefix in config.get('blocking_program_prefixes',[]):
  for name,status in states.items():
   if name.startswith(prefix) and status in ACTIVE:return False,'upstream workload active: '+name
 for item in config['prerequisites']:
  if item['program'] not in states:return False,'prerequisite supervisor entry unavailable'
  if states[item['program']] in ACTIVE:return False,'prerequisite program active: '+item['program']
  try:record=json.loads(Path(item['manifest']).read_text())
  except (OSError,ValueError):return False,'prerequisite manifest unavailable'
  jobs=record.get('jobs',[])
  if len(jobs)!=item['expected_jobs']:return False,'prerequisite job count mismatch'
  if any(j.get('status')=='failed' for j in jobs):return False,'prerequisite contains failed jobs; held for review'
  if not record.get('finished_unix') or any(j.get('status')!='completed' for j in jobs):
   return False,'prerequisite campaign still has pending jobs'
 return True,'prerequisites completed'

def gpu_inventory():
 def query(field):
  r=subprocess.run(['nvidia-smi',field,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
  if r.returncode:raise RuntimeError('GPU inventory unavailable: '+r.stderr)
  return list(csv.reader(r.stdout.strip().splitlines()))
 rows=query('--query-gpu=index,uuid,utilization.gpu,memory.used')
 inventory={int(r[0]):dict(uuid=r[1].strip(),utilization=int(r[2]),memory_mib=int(r[3]),pids=[]) for r in rows if r}
 for row in query('--query-compute-apps=gpu_uuid,pid'):
  if not row:continue
  if row[0].strip().lower().startswith('no running processes'):continue
  if len(row)<2:raise RuntimeError('Unrecognized compute-process inventory')
  for info in inventory.values():
   if row[0].strip()==info['uuid']:info['pids'].append(row[1].strip())
 return inventory

def gpu_idle(info,config):
 return not info['pids'] and info['memory_mib']<=config['max_idle_memory_mib'] and info['utilization']<=config['max_idle_utilization']

def completed(job,revision):
 try:
  p=Path(job['output']);done=json.loads((p/'completed.json').read_text());cfg=json.loads((p/'config.json').read_text())
  return done==dict(actor_updates=job['actor_updates'],completed=True) and cfg['baseline']==job['baseline'] and cfg['seed']==job['seed'] and cfg['steps']==job['actor_updates'] and cfg['runtime']['revision']==revision
 except (OSError,ValueError,KeyError):return False

def source_ready(config):
 def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True,stderr=subprocess.DEVNULL).strip()
 if git(config['repository_root'],'branch','--show-current')!=config['allowed_branch']:
  return False,'repository branch changed; dispatch held'
 if git(config['code_root'],'rev-parse','HEAD')!=config['revision']:return False,'pinned source revision changed'
 result=subprocess.run(['git','-C',config['code_root'],'diff','--quiet'],timeout=20)
 if result.returncode:return False,'pinned source has tracked modifications'
 return True,'source verified'

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--config',type=Path,required=True)
 parser.add_argument('--once',action='store_true')
 parser.add_argument('--dry-run',action='store_true')
 args=parser.parse_args();config=json.loads(args.config.read_text())
 state_path=Path(config['state_file'])
 if args.dry_run:state_path=state_path.with_name(state_path.stem+'.dry-run.json')
 lock=open(str(state_path)+'.lock','a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  lock.close();return
 try:old=json.loads(state_path.read_text())
 except (OSError,ValueError):old={}
 records={j['id']:dict(j,status='queued',reason='waiting for dependency checks') for j in config['jobs']}
 for item in old.get('jobs',[]):
  if item['id'] in records:records[item['id']].update(item)
 stable={g:0 for g in {j['gpu'] for j in config['jobs']}}
 held={};last_report=None
 while True:
  try:
   states=supervisor_states()
   for record in records.values():
    state=states.get(record['program'])
    if completed(record,config['revision']):
     record.update(status='completed',reason='completed marker and configuration verified')
     if 'finished_unix' not in record:record['finished_unix']=time.time()
     try:
      result=json.loads((Path(record['output'])/'evaluations'/f"step_{record['actor_updates']:07d}"/'metrics_full_policy.json').read_text())
      record['metrics']={k:result[k] for k in ('mmd','mode_coverage','sliced_wasserstein2')}
     except (OSError,ValueError,KeyError):pass
    elif state in ACTIVE:
     record.update(status='running',reason='managed by supervisor')
    elif record['status'] in ('running','starting') or state=='FATAL':
     record.update(status='failed',reason='training stopped without a valid completed marker')
    if record['status'] in ('completed','failed') and record['gpu'] in held and held[record['gpu']][0]==record['id']:
     handle=held[record['gpu']][1]
     fcntl.flock(handle,fcntl.LOCK_UN);handle.close();del held[record['gpu']]
   ready,reason=prerequisites_ready(config,states)
   if ready:ready,reason=source_ready(config)
   inventory=gpu_inventory() if ready else {}
   busy={r['gpu'] for r in records.values() if r['status'] in ('running','starting')}
   for gpu in sorted(stable):
    pending=[r for r in records.values() if r['gpu']==gpu and r['status']=='queued']
    if not pending:continue
    first=pending[0]
    if gpu in busy:
     stable[gpu]=0
     for r in pending:r['reason']='waiting for preceding queued GPU job'
     continue
    if not ready:
     stable[gpu]=0
     for r in pending:r['reason']=reason
     continue
    if gpu not in inventory or not gpu_idle(inventory[gpu],config):
     stable[gpu]=0
     for r in pending:r['reason']='GPU has compute processes, memory use, or activity'
     continue
    stable[gpu]+=1
    if stable[gpu]<config['required_idle_polls']:
     for r in pending:r['reason']='waiting for consecutive idle GPU checks'
     continue
    if first['program'] not in states:
     first['reason']='training supervisor entry unavailable';continue
    if Path(first['output']).exists():
     first.update(status='failed',reason='existing incomplete output is preserved; no retry');continue
    if args.dry_run:
     first['reason']='would start; dry-run prevents execution';continue
    lock_dir=Path(config['gpu_lock_directory']);lock_dir.mkdir(parents=True,exist_ok=True)
    gpu_lock=open(lock_dir/(inventory[gpu]['uuid']+'.lock'),'a')
    try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
     gpu_lock.close();first['reason']='GPU held by another queue';continue
    check,check_reason=prerequisites_ready(config,supervisor_states())
    if not check:
     fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();first['reason']=check_reason;continue
    try:
     check,check_reason=source_ready(config)
     fresh=gpu_inventory().get(gpu)
    except Exception:
     fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();raise
    if not check or fresh is None or not gpu_idle(fresh,config):
     fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();stable[gpu]=0
     first['reason']=check_reason if not check else 'GPU became occupied before dispatch';continue
    held[gpu]=(first['id'],gpu_lock);first.update(status='starting',reason='starting registered supervisor job',started_unix=time.time())
    atomic_json(state_path,dict(jobs=list(records.values()),status='dispatching',updated_unix=time.time()))
    result=subprocess.run(['sudo','-n','supervisorctl','start',first['program']],capture_output=True,text=True,timeout=30)
    current=supervisor_states().get(first['program'])
    if result.returncode==0 or current in ACTIVE:
     first.update(status='running',reason='supervisor started job')
    else:first.update(status='failed',reason='supervisor start failed: '+(result.stdout+result.stderr).strip())
    stable[gpu]=0
   terminal=all(r['status'] in ('completed','failed') for r in records.values())
   any_active=any(r['status'] in ('running','starting') for r in records.values())
   status=('completed_with_failures' if any(r['status']=='failed' for r in records.values()) else 'completed') if terminal else ('running' if any_active else 'waiting')
   snapshot=dict(campaign=config['campaign'],host=config['host'],pid=os.getpid(),status=status,
                 prerequisites_ready=ready,dependency_reason=reason,dry_run=args.dry_run,
                 updated_unix=time.time(),jobs=list(records.values()))
   atomic_json(state_path,snapshot)
   report=json.dumps(dict(status=status,dependency_reason=reason,jobs=[dict(id=r['id'],status=r['status'],reason=r['reason']) for r in records.values()]),sort_keys=True)
   if report!=last_report:print(report,flush=True);last_report=report
   if terminal or args.once:
    for _,handle in held.values():handle.close()
    lock.close();return
  except Exception as exc:
   print(json.dumps(dict(event='scheduler_error_dispatch_held',error=repr(exc))),flush=True)
   if args.once:
    for _,handle in held.values():handle.close()
    lock.close();raise
  time.sleep(config['poll_seconds'])

if __name__=='__main__':main()
