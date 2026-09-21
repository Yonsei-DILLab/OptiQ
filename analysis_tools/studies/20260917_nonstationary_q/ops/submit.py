"""Register all finite stage dependencies once, after all numerical validation."""
import json,subprocess,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parts='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'
def main():
    dest=root/'SUBMISSION.json';assert not dest.exists(),'Already submitted; inspect existing jobs first'
    code=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id']
    for i in range(4):
        v=json.loads((root/f'VALIDATION_{i}.json').read_text());assert v['passed'] and v['source_code_id']==code
    ts=json.loads((root/'tasks.json').read_text());previous=None
    record=dict(source_code_id=code,registered_at=datetime.datetime.now().astimezone().isoformat(),
        comparison_trajectories=448,execution_nodes=564,parallelism=16,cpus_per_gpu=2,jobs=[])
    for label,stages in [('prefix',['prefix']),('scripted',['mass','split']),('source',['source']),('replay',['replay']),('closed',['closed'])]:
        ids=[str(i) for i,t in enumerate(ts) if t['stage'] in stages]
        cmd=['sbatch','--parsable',f'--partition={parts}','--qos=big_qos',
             '--array='+','.join(ids)+'%16',f'--job-name=oq-nsq-{label}',
             f'--output={root}/logs/{label}-%A_%a.out']
        if previous:cmd+=['--dependency=afterok:'+previous]
        cmd+=[str(root/'job.sbatch')]
        result=subprocess.run(cmd,check=True,text=True,capture_output=True);job=result.stdout.strip().split(';')[0]
        assert job.isdigit(),result.stdout
        record['jobs'].append(dict(stage=label,job_id=job,count=len(ids),dependency=previous,command=cmd))
        previous=job;dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
