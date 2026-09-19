"""Bounded single-instance four-GPU pilot; failure stops that run, not the others."""
from pathlib import Path
import json,time,os,subprocess,sys,signal,fcntl,hashlib,socket

ROOT=Path('/home/heejoonorm/OptiQ/legacy_monge');PILOT=ROOT/'pilot';RUNTIME=ROOT/'runtime'
PY=Path('/home/heejoonorm/.venvs/optiq-monge/bin/python');STOP=False
def stopping(*_):
 global STOP
 STOP=True
def write(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 global STOP
 RUNTIME.mkdir(exist_ok=True)
 with (RUNTIME/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  dep=json.loads((PILOT/'DEPLOYMENT.json').read_text());checks=json.loads((PILOT/'validation/PASSED.json').read_text())
  assert checks['ok'] and checks['source_code_id']==dep['source_code_id']
  manifest=json.loads((PILOT/'SOURCE_MANIFEST.json').read_text())
  assert all(hashlib.sha256((PILOT/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
  ops=json.loads((ROOT/'OPERATIONS_DEPLOYMENT.json').read_text())
  tasks=json.loads((PILOT/'tasks.json').read_text());run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
  record=dict(run_id=run_id,host=socket.gethostname(),pid=os.getpid(),started=time.time(),
   numerical_commit=dep['commit'],operations_commit=ops['commit'],source_code_id=dep['source_code_id'],
   gpu_ids=[0,1,2,3],max_wall_seconds=7200,events=[])
  out=RUNTIME/(run_id+'.json');write(out,record)
  for s in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(s,stopping)
  allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=8
  pending=[i for i,t in enumerate(tasks) if not (PILOT/'runs'/t['name']/'COMPLETE.json').exists()]
  active={};deadline=time.monotonic()+7200
  while pending or active:
   if STOP or time.monotonic()>=deadline:
    pending=[]
    for gpu,(p,f,ix,start) in active.items():
     if p.poll() is None:p.send_signal(signal.SIGUSR1)
    for gpu,(p,f,ix,start) in active.items():
     try:code=p.wait(timeout=120)
     except subprocess.TimeoutExpired:p.terminate();code=p.wait(timeout=30)
     f.close();record['events'].append(dict(event='checkpoint_stop',index=ix,gpu=gpu,code=code,time=time.time()))
    active={};break
   for gpu in list(active):
    p,f,ix,start=active[gpu];code=p.poll()
    if code is None:continue
    f.close();del active[gpu]
    record['events'].append(dict(event='exit',index=ix,gpu=gpu,code=code,time=time.time(),elapsed=time.time()-start))
    print('EXIT',tasks[ix]['name'],code,flush=True);write(out,record)
   for gpu in [0,1,2,3]:
    if gpu in active or not pending:continue
    ix=pending.pop(0);task=tasks[ix];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPTIQ_RUN_ID=run_id)
    f=(RUNTIME/(task['name']+'_'+run_id+'.log')).open('a')
    cpus=allowed[gpu*2:gpu*2+2]
    p=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),str(PY),'experiment.py','--index',str(ix)],cwd=PILOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    active[gpu]=(p,f,ix,time.time());record['events'].append(dict(event='start',index=ix,name=task['name'],gpu=gpu,cpus=cpus,pid=p.pid,time=time.time()))
    write(out,record);print('START',task['name'],'gpu',gpu,'pid',p.pid,flush=True)
   time.sleep(2)
  record['ended']=time.time();record['completed']=sum((PILOT/'runs'/t['name']/'COMPLETE.json').exists() for t in tasks)
  record['state']='complete' if record['completed']==len(tasks) else 'incomplete';write(out,record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()
