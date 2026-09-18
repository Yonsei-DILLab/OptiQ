"""One allocated GPU: claim ready independent tasks; wait only for true data dependencies."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,sys,time,socket
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nsq.config import write_json
STOP=False;child=None

def stop(*_):
 global STOP
 STOP=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGUSR1)

for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,stop)

def main():
 global child
 plan=json.loads((ROOT/'tasks.json').read_text());code=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['code_id']
 for d in [1,2,4,8]:
  v=json.loads((ROOT/f'VALIDATION_D{d}.json').read_text());assert v['passed'] and v['source_code_id']==code
 (ROOT/'queue').mkdir(exist_ok=True);job=(os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_JOB_ID' in os.environ else os.environ.get('SLURM_JOB_ID','local')
 idle=0
 while not STOP:
  with (ROOT/'queue/claim.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   result=subprocess.run(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],text=True,capture_output=True,check=True)
   active=set(result.stdout.split());ready=[];pending=0
   for i,t in enumerate(plan):
    out=ROOT/'runs'/t['name'];lease=ROOT/'queue'/(t['name']+'.json')
    if (out/'COMPLETE.json').exists() or (out/'FAILED.json').exists():continue
    pending+=1
    if lease.exists():
     previous=json.loads(lease.read_text())
     # Exact concrete job IDs are written by workers, and squeue -r expands arrays.
     if previous['job_id'] in active:continue
     # Do not reclaim a lease younger than one minute during scheduler propagation.
     if time.time()-previous['time']<60:continue
     lease.unlink()
    deps=[x for x in [t.get('parent'),t.get('q_source')] if x]
    if all((ROOT/'runs'/x/'COMPLETE.json').exists() for x in deps):ready.append((i,t,lease))
   if ready:
    i,t,lease=ready[0];write_json(lease,dict(job_id=job,time=time.time(),task_index=i,host=socket.gethostname()))
   else:i=None
  if i is None:
   if pending==0:print('All runnable trials processed.',flush=True);return
   idle+=1
   if idle>=4:
    print('No independently ready work; releasing GPU. Later queue wave can resume.',flush=True);return
   time.sleep(15);continue
  idle=0;out=ROOT/'runs'/t['name'];out.mkdir(parents=True,exist_ok=True)
  print('Starting',i,t['name'],flush=True)
  with (out/'console.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-m','nsq.run','--root',str(ROOT),'--index',str(i)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   code=child.wait();child=None
  with (ROOT/'queue/claim.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if lease.exists():lease.unlink()
  if code and code!=75 and not (out/'FAILED.json').exists():write_json(out/'FAILED.json',dict(returncode=code,time=time.time(),job_id=job))
  if code==75 or STOP:return
if __name__=='__main__':main()
