import concurrent.futures,datetime,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=BASE/'snapshots'/stamp;out.mkdir(parents=True,exist_ok=True)
script=(BASE/'remote_snapshot.py').read_text()
def collect(host):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -'],input=script,text=True,capture_output=True,check=True)
 d=json.loads(p.stdout);d['host']=host;(out/(host+'.json')).write_text(json.dumps(d,indent=2))
 summary={'host':host,'learner_count':len(d['learners']),'campaigns':{}}
 for c,v in d['campaigns'].items():
  jobs=v['metadata'].get('queue.json',{}).get('jobs',list(v['jobs'].values()))
  count={}
  for j in jobs:count[j['state']]=count.get(j['state'],0)+1
  rows=[]
  for n,p in v['progress'].items():
   e=p.get('latest_evaluation',{});mode=e.get('primary_evaluation_mode','policy');metric=e.get(mode,{})
   rows.append({'name':n,'state':p['status'],'steps':p['steps'],'updates':p['updates'],'success':metric.get('success'),'goals':metric.get('goals')})
  summary['campaigns'][c]={'queue_state':v['metadata'].get('queue.json',{}).get('state'),'counts':count,'rows':rows,'failures':v['failures'],'proofs':len(v['proofs'])}
 return summary
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(collect,['vast-heechan-46','vast-heechan-199','vast-heechan-6']))
(out/'summary.json').write_text(json.dumps(results,indent=2));(BASE/'latest-snapshot.txt').write_text(str(out))
print(json.dumps(results,indent=2));print(out)
