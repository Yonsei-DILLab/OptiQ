"""Add only the newly authorized port41188 instance to central collection."""
from pathlib import Path
import subprocess,json,tarfile
P=Path(__file__).resolve().parent;OLD=P.parent.parent/'fetch-adroit-f1-a1-temp-20261112'
central='/data1/heechan/iBOLT/20261112-f1-a1-temp'
host2="""from central_collector import *
while True:
 try:
  result=once(2,initial=False)
  if result['passed']:
   base=ROOT/'hosts/host2/workspace'/NEW
   proof=json.loads((base/'transfer-proof.json').read_text())
   assert all(digest(base/name)==sha for name,sha in proof.items())
   remote(2,'archive-ready',dict(passed=True,hashes=proof,central=str(base),verified_unix=time.time()))
 except Exception as e:
  (ROOT/'host2-archive-error.json').write_text(json.dumps(dict(at=time.time(),error=repr(e))))
 time.sleep(45)
"""
(P/'central_host2.py').write_text(host2)
kh=subprocess.check_output(['ssh-keygen','-F','[114.32.64.6]:41188'],text=True);kh='\n'.join(l for l in kh.splitlines() if not l.startswith('#'))+'\n'
code='KNOWN='+repr(kh)+'\n'+'''from pathlib import Path
import shutil,time
p=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');shutil.copy2(p/'central_collector.py',p/('central_collector.before-host2-'+str(int(time.time()))+'.py'))
f=Path.home()/'.ssh/ibolt_archive_known_hosts';s=f.read_text();f.write_text(s.rstrip()+'\\n'+KNOWN)
print('backup and host key ready')
'''
ssh=['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr']
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=25);assert r.returncode==0,r.stderr
with tarfile.open(P/'central-host2.tgz','w:gz') as t:
 t.add(OLD/'central_collector.py',arcname='central_collector.py');t.add(P/'central_host2.py',arcname='central_host2.py')
with (P/'central-host2.tgz').open('rb') as f:
 r=subprocess.run(ssh+['tar -xz -C '+central],stdin=f,capture_output=True,timeout=25);assert r.returncode==0,r.stderr
code=r'''
from pathlib import Path
import os,subprocess,signal,json,shlex,time
p=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');os.chdir(p)
from central_collector import remote
assert json.loads(remote(2,'checkpoint-manifest'))==[]
stopped=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if d.stat().st_uid!=os.getuid() or (d/'cwd').resolve()!=p:continue
  args=(d/'cmdline').read_bytes().split(b'\0');args=[a.decode() for a in args if a]
  if not args or Path(args[0]).name not in ['python3','python']:continue
  if any(Path(a).name in ['central_live.py','central_entropy.py'] for a in args[1:]):
   os.kill(int(d.name),signal.SIGTERM);stopped.append(int(d.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for script,session in [('central_live.py','ibolt-central-live-host2-20261112'),('central_entropy.py','ibolt-central-entropy-host2-20261112'),('central_host2.py','ibolt-central-host2-20261112')]:
 command='cd '+str(p)+' && python3 -u '+script+' >> '+script.removesuffix('.py')+'-host2.log 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
print(json.dumps(dict(host2_read_access_verified=True,old_coordinator_pids=stopped,new_collectors_started=True)))
'''
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=30);assert r.returncode==0,r.stdout+r.stderr
(P/'central-expanded.json').write_text(r.stdout);print(r.stdout)
