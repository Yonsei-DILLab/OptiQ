"""Immutable revision: validation array -> independently claiming GPU workers."""
from pathlib import Path
import argparse,json,subprocess,sys,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from nsq.config import write_json
parts='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','main']);args=ap.parse_args()
code=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id'];commit=json.loads((root/'DEPLOYMENT.json').read_text())['commit']
dest=root/('VALIDATION_SUBMISSION.json' if args.mode=='validate' else 'SUBMISSION.json');assert not dest.exists()
(root/'logs').mkdir(exist_ok=True)
if args.mode=='main':
 for d in [1,2,4,8]:
  v=json.loads((root/f'VALIDATION_D{d}.json').read_text());assert v['passed'] and v['source_code_id']==code
if args.mode=='validate':parts=','.join(p for p in parts.split(',') if p!='big_suma_rtx3090')
cmd=['sbatch','--parsable','--partition='+parts,'--qos='+('base_qos' if args.mode=='validate' else 'big_qos'),
 '--array='+('0-3%4' if args.mode=='validate' else '0-15%16'),
 '--job-name=nsq-nd-'+args.mode,'--output='+str(root/'logs'/f'{args.mode}-%A_%a.out')]
if args.mode=='validate':cmd+=['--time=01:00:00']
cmd += [str(root/'job.sbatch'),str(root),'validate' if args.mode=='validate' else 'worker']
write_json(dest,dict(state='submitting',commit=commit,source_code_id=code,command=cmd,time=time.time()))
r=subprocess.run(cmd,check=True,text=True,capture_output=True);job=r.stdout.strip().split(';')[0];assert job.isdigit()
write_json(dest,dict(state='submitted',commit=commit,source_code_id=code,job_id=job,command=cmd,time=time.time()))
print(dest.read_text())
