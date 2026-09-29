"""Read-only snapshots of the two authorized 16-run campaign shards."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,datetime,json,subprocess,sys
BASE=Path(__file__).resolve().parent
ROOT='/home/heechan/optiq-experiments/antmaze-dense-noveld-1m-s0-20260922'
CODE='''from pathlib import Path
import json,subprocess
p=Path("%s")
result={}
for n in ["manifest.json","status.json","failure.json","result.json","registration.json"]:
 if (p/n).exists():result[n]=json.loads((p/n).read_text())
result["runs"]={}
for f in sorted((p/"runs").glob("*")):
 d={}
 for n in ["config.json","progress.json","wandb.json","checkpoint.json","failure.json","result.json","verification.json"]:
  if (f/n).exists():d[n]=json.loads((f/n).read_text())
 result["runs"][f.name]=d
result["proofs"]={f.stem:json.loads(f.read_text()) for f in (p/"proofs").glob("*.json")}
result["processes"]={}
for j in result.get("status.json",{}).get("jobs",[]):
 if j.get("status")=="running":
  r=subprocess.run(["ps","-p",str(j["pid"]),"-o","pid=,etime=,args="],capture_output=True,text=True)
  result["processes"][j["id"]]=r.stdout.strip()
print(json.dumps(result))
'''%ROOT

def collect(host):
 r=subprocess.run(['ssh',host,'python3','-'],input=CODE,text=True,capture_output=True,check=True,timeout=60)
 d=json.loads(r.stdout)
 (BASE/(host+'.json')).write_text(json.dumps(d,indent=2)+'\n')
 s=d.get('status.json',{});print(host,s.get('stage'),s.get('phase'),s.get('completed'),s.get('total'))
 for name,v in d['runs'].items():print(name,v.get('progress.json',{}).get('env_steps'),v.get('wandb.json',{}).get('url'))
 return host,d
if __name__=='__main__':
 parser=argparse.ArgumentParser(allow_abbrev=False)
 parser.add_argument('--archive-completed',action='store_true')
 args=parser.parse_args()
 with ThreadPoolExecutor(2) as ex:d=dict(ex.map(collect,['vast-heechan-180','vast-heechan-199']))
 (BASE/'latest.json').write_text(json.dumps(dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=d),indent=2)+'\n')
 if args.archive_completed:
  sys.path.insert(0,str(BASE.parents[1]))
  from antmaze.multimodal.dense_noveld_report import verify_run,write,TRAINING_SHA
  for host,data in d.items():
   assert data['manifest.json']['source_commit']==TRAINING_SHA
   for j in data['status.json']['jobs']:
    if j['status']!='completed':continue
    name=j['id'];assert name==f"{j['task']}-{j['method']}-s0"
    assert j['task'] in ['v1','v2','v3','v4'] and j['method'] in ['optiq','sac','meow','mfpo']
    dest=BASE/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    files=['config.json','result.json','verification.json','progress.json','wandb.json','checkpoint.json',
     'parameter-audit.json','intrinsic-audit.json','training_coverage.npz','training_episodes.json',
     'history-native-natural.json','history-policy-natural.json']
    filters=[f'--include=/{n}' for n in files]+['--include=/rollouts/***','--include=/resume/',
     '--include=/resume/step_0001000000/***','--exclude=*']
    subprocess.run(['rsync','-a','--checksum',*filters,host+':'+ROOT+'/runs/'+name+'/',str(dest)+'/'],check=True)
    proof=verify_run(dest)
    write(dest/'archive-verification.json',dict(**proof,host=host,remote=ROOT+'/runs/'+name,
      archived_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ARCHIVED AND VERIFIED',name,flush=True)
