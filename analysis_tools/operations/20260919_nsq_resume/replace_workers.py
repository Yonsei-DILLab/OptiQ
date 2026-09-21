"""One replacement per original worker, without interrupting healthy training."""
from pathlib import Path
import argparse,json,subprocess,time
from worker import scheduler_snapshot,write

def main(ops,original,event):
    ops=Path(ops).resolve();old=Path(original).resolve()
    active=scheduler_snapshot();assert active is not None
    source=json.loads((old/'SUBMISSIONS.json').read_text())
    record=ops/'REPLACEMENTS.json';rows=json.loads(record.read_text()) if record.exists() else []
    submitted={r['original_worker'] for r in rows};(ops/'logs').mkdir(exist_ok=True)
    for group in source:
        mode=group['mode']
        if mode not in ['main','legacy']:continue
        count=16 if mode=='main' else 4
        root=Path(group['root']);parts=next(x for x in group['command'] if x.startswith('--partition='))
        for i in range(count):
            previous=group['job_id']+'_'+str(i)
            if previous in submitted:continue
            dependency='afterany:'+previous if previous in active else None
            cmd=['sbatch','--parsable',parts,'--qos=big_qos','--array=0-0%1','--job-name=nsq-resume-'+mode,'--output='+str(ops/'logs'/(mode+'-%A_%a.out'))]
            if dependency:cmd.append('--dependency='+dependency)
            cmd += [str(ops/'job.sbatch'),str(root),str(ops),mode,event]
            result=subprocess.run(cmd,capture_output=True,text=True,check=True);job=result.stdout.strip().split(';')[0];assert job.isdigit()
            row=dict(original_worker=previous,replacement_job=job,mode=mode,dependency=dependency,root=str(root),time=time.time(),command=cmd,operations_commit=json.loads((ops/'OPS_DEPLOYMENT.json').read_text())['commit'],numerical_commit=json.loads((root/'DEPLOYMENT.json').read_text())['commit'])
            rows.append(row);write(record,rows);print(json.dumps({k:row[k] for k in ['original_worker','replacement_job','dependency']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ops',required=True);p.add_argument('--original',required=True);p.add_argument('--event',required=True);a=p.parse_args();main(a.ops,a.original,a.event)
