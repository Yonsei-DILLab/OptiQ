from pathlib import Path
import subprocess,json,concurrent.futures,shlex
P=Path(__file__).resolve().parent;root='/workspace/heechan-unified-f1-a1-20261112'
pub='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMi26dumpQqk58iMoZwWKR+/4UloJBzGd65UJoRugiHu ibolt-central-archive-20261112'
gateway=(P/'archive_gateway.py').read_text()
remote=r'''
from pathlib import Path
import os,json
base=Path(ROOT);base.mkdir(exist_ok=True)
(base/'archive_gateway.py').write_text(GATEWAY);(base/'archive_gateway.py').chmod(0o700)
roots=[p.name for p in Path('/workspace').iterdir() if p.is_dir() and not p.is_symlink() and p.stat().st_uid==os.getuid() and (p.name.startswith('heechan-') or p.name in ('iBOLT','iBOLT-runs'))]
# Historical cleanup is limited to the user's Adroit result trees and old Fetch exploration outputs.
checkpoint_roots=['heechan-adroit']
if HOST==3:checkpoint_roots+=['heechan-fetch-dacer/outputs']
checkpoint_roots+=[base.name+'/runs']
(base/'runs').mkdir(exist_ok=True)
config=dict(allowed_roots=roots,checkpoint_roots=checkpoint_roots,central='dil.yonsei.ac.kr:/data1/heechan/iBOLT/20261112-f1-a1-temp')
(base/'archive-access.json').write_text(json.dumps(config,indent=2))
ssh=Path.home()/'.ssh';ssh.mkdir(mode=0o700,exist_ok=True);auth=ssh/'authorized_keys'
line='restrict,command="/usr/bin/python3 '+str(base/'archive_gateway.py')+'" '+PUB
text=auth.read_text() if auth.exists() else ''
if PUB not in text:
 with auth.open('a') as f:f.write(('\n' if text and not text.endswith('\n') else '')+line+'\n')
auth.chmod(0o600)
print(json.dumps(config))
'''
def one(n):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{n}','python3 -'],input='ROOT='+repr(root)+'\nHOST='+repr(n)+'\nPUB='+repr(pub)+'\nGATEWAY='+repr(gateway)+'\n'+remote,text=True,capture_output=True,timeout=30)
 assert not r.returncode,r.stderr;(P/f'archive-host{n}.json').write_text(r.stdout);return n,'configured'
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
 for r in e.map(one,[3,4,5]):print(r)
# Trust only the host keys already used and verified by the local SSH client.
known=Path.home()/'.ssh/known_hosts'
lines=[]
for hostport in ['[211.72.13.201]:42866','[175.155.64.199]:19280','[211.72.13.201]:42757']:
 r=subprocess.run(['ssh-keygen','-F',hostport,'-f',str(known)],text=True,capture_output=True,check=True)
 lines.extend(x for x in r.stdout.splitlines() if not x.startswith('#'))
code='from pathlib import Path\np=Path.home()/".ssh/ibolt_archive_known_hosts";p.write_text('+repr('\n'.join(lines)+'\n')+');p.chmod(0o600)\np=Path("/data1/heechan/iBOLT/20261112-f1-a1-temp");p.mkdir(parents=True,exist_ok=True)\n'
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code,text=True,capture_output=True,timeout=30);assert not r.returncode,r.stderr
