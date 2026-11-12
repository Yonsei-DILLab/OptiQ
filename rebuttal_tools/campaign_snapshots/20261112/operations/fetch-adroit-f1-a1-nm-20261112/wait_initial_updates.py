from pathlib import Path
import sys,subprocess,json,time
P=Path(__file__).resolve().parent
for i in range(12):
 r=subprocess.run([sys.executable,str(P/'status.py')],text=True,capture_output=True,timeout=70)
 assert r.returncode==0,r.stderr
 d=json.loads(r.stdout)
 print(json.dumps({k:d[k] for k in ['at','seen','counts','verified','updating']}),flush=True)
 assert all(not h['controller_failure'] and not h['failures'] for h in d['hosts']),d
 if d['seen']==d['verified']==d['updating']==32:
  (P/'initial-updates-verified.json').write_text(json.dumps(d,indent=2));break
 time.sleep(20)
else:raise RuntimeError('Initialization not complete; inspect status and logs')
