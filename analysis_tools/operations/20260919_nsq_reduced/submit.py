"""Idempotent, audited independent worker registration after four-dimensional validation."""
from pathlib import Path
import argparse,json,subprocess,time
PARTITIONS='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'
def launch(root,ops):
 code=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id']
 for dim in (1,2,4,8):
  for gate in ('VALIDATION','LEGACY_TD_VALIDATION'):
   r=json.loads((root/f'{gate}_D{dim}.json').read_text());assert r['passed'] and r['source_code_id']==code
 tasks=json.loads((root/'tasks.json').read_text());assert len(tasks)==2912
 assert {(t['n'],t['m']) for t in tasks}=={(256,1024),(512,512)}
 record=ops/'SUBMISSIONS.json';rows=json.loads(record.read_text()) if record.exists() else []
 (ops/'logs').mkdir(exist_ok=True)
 for lane,count in [('fast',18),('exact',2)]:
  if any(r['lane']==lane for r in rows):continue
  cmd=['sbatch','--parsable','--partition='+PARTITIONS,'--qos=big_qos',f'--array=0-{count-1}%{count}',
       '--job-name=nsq-small-'+lane,'--output='+str(ops/'logs'/(lane+'-%A_%a.out')),
       str(ops/'job.sbatch'),str(root),str(ops),lane,'reduced_20260919']
  result=subprocess.run(cmd,capture_output=True,text=True,check=True);job=result.stdout.strip().split(';')[0];assert job.isdigit()
  rows.append(dict(lane=lane,workers=count,job_id=job,time=time.time(),command=cmd,commit=json.loads((root/'DEPLOYMENT.json').read_text())['commit'],source_code_id=code))
  tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(record)
  print(json.dumps(rows[-1]),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--ops',required=True);a=p.parse_args();launch(Path(a.root),Path(a.ops))
