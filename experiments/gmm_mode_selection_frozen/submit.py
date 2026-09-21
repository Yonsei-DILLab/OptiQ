import json,subprocess,time
from pathlib import Path
from experiments.gmm_gradient_interference import bootstrap
ROOT=bootstrap.ROOT;P=Path(__file__).parent;plan=json.loads((P/'plan.json').read_text());out=ROOT/'runtime'
out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
record=out/'SUBMISSION.json';assert not record.exists(),'already submitted'
rec=dict(time=time.time(),commit=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['commit'],jobs=[])
def submit(stage,args):
 cmd=['sbatch','--parsable','--job-name=mode-frozen-b128-'+stage,'--output='+str(out/'logs'/(stage+'-%A_%a.log'))]+args+[str(P/'job.sbatch'), 'validate' if stage=='validation' else 'worker']
 res=subprocess.run(cmd,text=True,capture_output=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit()
 rec['jobs'].append(dict(stage=stage,id=jid,command=cmd));record.write_text(json.dumps(rec,indent=2)+'\n');return jid
val=submit('validation',['--partition=dell_cpu','--qos=cpu_qos','--gres=gpu:0','--mem=8G','--time=00:20:00'])
gate=['--dependency=afterok:'+val,'--kill-on-invalid-dep=yes']
submit('general',['--partition='+plan['partitions'],'--qos='+plan['qos'],'--exclude='+plan['excluded_nodes']]+gate+['--array=0-'+str(plan['general_workers']-1)])
print(record.read_text())
