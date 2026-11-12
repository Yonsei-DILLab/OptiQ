from pathlib import Path
import subprocess,json,shlex
P=Path(__file__).resolve().parent
known=subprocess.run(['ssh-keygen','-F','[122.117.118.68]:41261','-f',str(Path.home()/'.ssh/known_hosts')],capture_output=True,text=True,check=True)
lines=[l for l in known.stdout.splitlines() if l and not l.startswith('#')];assert lines
payload={n:(P/n).read_text() for n in ['central_collector.py','central_host1.py']}
code='FILES='+repr(payload)+'\nKNOWN='+repr('\n'.join(lines)+'\n')+'\n'+r'''
from pathlib import Path
import subprocess,json,shlex,time
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp')
known=Path.home()/'.ssh/ibolt_archive_known_hosts';old=known.read_text()
if KNOWN not in old:known.write_text(old+KNOWN)
for n,body in FILES.items():
 p=root/n
 if p.exists() and not (root/(n+'.before-host1')).exists():(root/(n+'.before-host1')).write_bytes(p.read_bytes())
 p.write_text(body)
session='ibolt-host1-archive-20261112'
r=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
if r.returncode:subprocess.run(['tmux','new-session','-d','-s',session,'cd '+shlex.quote(str(root))+' && python3 -u central_host1.py >> host1-archive.log 2>&1'],check=True)
# The metadata/selector daemon is safe to replace independently of training and archive workers.
cmd='cd '+shlex.quote(str(root))+' && python3 -u central_live.py >> live.log 2>&1'
subprocess.run(['tmux','respawn-window','-k','-t','ibolt-live-20261112:0',cmd],check=True)
print('CENTRAL_HOST1_ENABLED')
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code,text=True,capture_output=True,timeout=30);assert not r.returncode,r.stdout+r.stderr;print(r.stdout)
