import json,subprocess,concurrent.futures,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
CODE="""import json,time
from pathlib import Path
r=Path('/home/heechan/optiq-experiments/antmaze-matched-start-eval-20260924-r2')
d={'time':time.time()}
for n in ['manifest','registration','status','failure','result']:
 p=r/(n+'.json');d[n]=json.loads(p.read_text()) if p.exists() else None
d['workers']=[]
for item in (d['status'] or {}).get('running',[]):
 out=Path(item['output']);row={'job':item['job'],'step':item['step']}
 for n in ['progress','result','verification','wandb-failure']:
  p=out/(n+'.json');row[n]=json.loads(p.read_text()) if p.exists() else None
 log=r/'logs'/(item['job']+'-'+str(item['step'])+'.log')
 row['log_tail']=log.read_text()[-1800:] if log.exists() else ''
 d['workers'].append(row)
print(json.dumps(d))
"""
def collect(h):
 p=subprocess.run(['ssh',h,'python3','-'],input=CODE,text=True,capture_output=True,check=True)
 d=json.loads(p.stdout);(OUT/(h+'.json')).write_text(json.dumps(d,indent=2))
 return {'host':h,**d}
with concurrent.futures.ThreadPoolExecutor(3) as pool:rows=list(pool.map(collect,['vast-heechan-180','vast-heechan-199','vast1']))
(OUT/'latest-status.json').write_text(json.dumps({'collected':time.time(),'hosts':rows},indent=2))
for h in rows:
 s=h['status'] or {};print(h['host'],json.dumps({k:s.get(k) for k in ['running','pending','completed','failed']}))
 for w in h['workers']:print(w)
