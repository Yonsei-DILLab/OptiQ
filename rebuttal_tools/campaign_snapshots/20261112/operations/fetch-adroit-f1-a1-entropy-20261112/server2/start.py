from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
assert json.loads((P/'committed.json').read_text())['passed']
code=r"""
from pathlib import Path
import json,subprocess,time
p=Path('/workspace/heechan-unified-f1-a1-20261112');d=json.loads((p/'plan.json').read_text())
assert len(d['jobs'])==55 and all(not j.get('pid') for j in d['jobs'])
for name in ['preflight.json','archive-ready.json','entropy-archive-ready.json']:assert json.loads((p/name).read_text())['passed']
assert json.loads((p/'slide-20cm-20261112/environment-verification.json').read_text())['passed']
assert d['python']['adroit']=='/workspace/heechan-adroit/venv/bin/python'
name='ibolt-f1-a1-temp-20261112-host2'
conf=f'''[program:{name}]
command=/usr/bin/python3 -u {p}/controller.py
directory={p}
user=heechan
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={p}/controller.log
stdout_logfile_maxbytes=10485760
'''
target=Path('/etc/supervisor/conf.d')/(name+'.conf');assert not target.exists()
subprocess.run(['sudo','-n','tee',str(target)],input=conf,text=True,stdout=subprocess.DEVNULL,check=True)
subprocess.run(['sudo','-n','supervisorctl','reread'],capture_output=True,check=True)
subprocess.run(['sudo','-n','supervisorctl','update',name],capture_output=True,check=True)
subprocess.run(['sudo','-n','supervisorctl','start',name],capture_output=True,check=True)
assert (p/'PAUSE').read_text().startswith('server2 bootstrap')
(p/'PAUSE').unlink();time.sleep(3)
print(json.dumps(dict(passed=True,host=2,controller=subprocess.check_output(['sudo','-n','supervisorctl','status',name],text=True).strip(),jobs=len(d['jobs']),gpu_limit=5)))
"""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-2','python3 -'],input=code,text=True,capture_output=True,timeout=45)
assert r.returncode==0,r.stdout+r.stderr
(P/'started.json').write_text(r.stdout);print(r.stdout)
