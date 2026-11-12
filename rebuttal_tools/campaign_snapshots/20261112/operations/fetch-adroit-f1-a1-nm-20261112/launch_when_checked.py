from pathlib import Path
import sys,json,time,subprocess
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent
for attempt in range(15):
 r=ssh(2,'from pathlib import Path\nimport json\np=Path('+repr(ROOT+'/nm-ablation-20261112/numeric-fetch.log')+');print(json.dumps({"log":p.read_text()}))')
 lines=[s for s in r['log'].splitlines() if s.startswith('{"passed":')]
 if lines:
  d=json.loads(lines[-1]);assert d['passed'];(P/'numeric-fetch.json').write_text(json.dumps(d,indent=2));print('FETCH NUMERIC PASSED',flush=True);break
 assert 'Traceback' not in r['log'],r['log'][-3000:]
 print('GPU numerical compilation continues',flush=True);time.sleep(15)
else:raise RuntimeError('GPU validation did not complete; no experiment admitted')
subprocess.run([sys.executable,str(P/'admit.py')],check=True)
