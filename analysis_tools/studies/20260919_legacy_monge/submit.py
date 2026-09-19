"""Submit validation first, then a bounded independent pilot array after it passes."""
from pathlib import Path
import argparse,subprocess,json,time
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','run']);a=p.parse_args()
root=Path(__file__).resolve().parent;dep=json.loads((root/'DEPLOYMENT.json').read_text());(root/'logs').mkdir(exist_ok=True)
if a.mode=='run':
 passed=json.loads((root/'validation'/'PASSED.json').read_text());assert passed['ok'] and passed['source_code_id']==dep['source_code_id']
parts='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'
cmd=['sbatch','--parsable','-p',parts,'-q','big_qos','--job-name=monge-'+a.mode,
 '--output='+str(root/'logs'/'%A_%a.out')]
if a.mode=='run':cmd+=['--array=0-11%4']
cmd+=[str(root/'job.sbatch'),str(root),a.mode]
r=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
record=dict(mode=a.mode,command=cmd,commit=dep['commit'],source_code_id=dep['source_code_id'],time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
f=root/'SUBMISSIONS.json';rows=json.loads(f.read_text()) if f.exists() else [];rows.append(record);f.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(record,indent=2));r.check_returncode()
