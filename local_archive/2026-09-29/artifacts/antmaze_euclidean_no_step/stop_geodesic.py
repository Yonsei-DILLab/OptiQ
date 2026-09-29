import json,os,subprocess,time
from pathlib import Path
import psutil
BASE=Path('/home/heechan/optiq-experiments');OPS=Path('/home/heechan/OptiQ-ops')
ROOTS=[BASE/x for x in ['antmaze-optiq-geodesic-no-step-B0-T1-s0-20260924','antmaze-optiq-progress100-2x2-T1-s0-20260924'] if (BASE/x).exists()]
REASON='User stopped all geodesic experiments and requested fresh Euclidean progress-only B0/step-cost0 runs.'
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def matching():
 matches=[]
 for p in psutil.process_iter(['pid','cmdline','create_time','status']):
  if p.pid==os.getpid():continue
  try:
   cmd=p.info['cmdline'] or []
   if any(any(str(r) in arg for r in ROOTS) for arg in cmd) and p.info['status']!=psutil.STATUS_ZOMBIE:matches.append(p)
  except psutil.NoSuchProcess:pass
 return matches
before={str(r):{f:json.loads((r/f).read_text()) for f in ['manifest.json','status.json','failure.json'] if (r/f).exists()} for r in ROOTS}
processes=[dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline()) for p in matching()]
for r in ROOTS:
 assert not (r/'cancelled-for-euclidean-no-step-20260925.json').exists()
 dump(r/'before-euclidean-no-step-stop-20260925.json',dict(snapshot=before[str(r)],processes=processes,time=time.time()))
ctl=['/usr/local/bin/supervisorctl','-c',str(OPS/'supervisor/supervisord.conf')]
services=[]
for r in ROOTS:services += [r.name,r.name+'-geodesic-only',r.name+'-wandb-sync']
services+=['antmaze-matched-start-eval-20260924-r2']
receipts=[]
for name in services:
 conf=OPS/'supervisor/jobs'/f'{name}.conf'
 if not conf.exists():continue
 text=conf.read_text();conf.write_text(text.replace('autostart=true','autostart=false').replace('autorestart=true','autorestart=false'))
 p=subprocess.run(ctl+['stop',name],text=True,capture_output=True,timeout=45)
 receipts.append(dict(service=name,code=p.returncode,output=p.stdout.strip(),error=p.stderr.strip()))
# Adopted wrappers may not be children of their monitoring controller.
victims={}
for parent in matching():
 try:
  for p in parent.children(recursive=True)+[parent]:victims[p.pid]=p
 except psutil.NoSuchProcess:pass
for p in victims.values():
 try:p.terminate()
 except psutil.NoSuchProcess:pass
_,live=psutil.wait_procs(list(victims.values()),timeout=12)
for p in live:
 try:p.kill()
 except psutil.NoSuchProcess:pass
psutil.wait_procs(live,timeout=5)
remaining=[dict(pid=p.pid,cmdline=p.cmdline()) for p in matching()];assert not remaining,remaining
summary=[]
for r in ROOTS:
 m=json.loads((r/'manifest.json').read_text());cancelled=[];completed=[]
 for entry in m['jobs']:
  if 'geodesic' not in entry.get('reward_profile',''):continue
  key=entry['id'];jp=r/'jobs'/f'{key}.json';old=json.loads(jp.read_text()) if jp.exists() else dict(entry)
  result=r/'runs'/key/'result.json'
  if result.exists() and json.loads(result.read_text()).get('completed'):
   completed.append(key);continue
  if old.get('status') in ('cancelled','interrupted','failed'):continue
  dump(r/'jobs'/f'{key}-before-user-stop-20260925.json',old)
  dump(jp,dict(**{k:v for k,v in old.items() if k!='status'},status='cancelled',cancelled_at=time.time(),cancel_reason=REASON))
  cancelled.append(key)
 state=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {}
 state.update(controller_pid=None,running=[],pending=[],cancelled=sorted(set(state.get('cancelled',[])+cancelled)),pending_held=True,stopped_by_user=True,stopped_at=time.time())
 dump(r/'status.json',state)
 record=dict(reason=REASON,time=time.time(),cancelled=cancelled,completed_preserved=completed,services=receipts,remaining_processes=remaining,source_manifest_unchanged=True,no_automatic_resume=True)
 dump(r/'cancelled-for-euclidean-no-step-20260925.json',record);summary.append(dict(root=str(r),cancelled=cancelled,completed_preserved=completed))
print(json.dumps(dict(summary=summary,remaining=remaining,services=receipts)))
