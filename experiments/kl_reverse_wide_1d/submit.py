import argparse,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['validate','train'],required=True)
    a=p.parse_args();r=a.root.resolve();rt=r/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    record_path=rt/('VALIDATION_SUBMISSION.json' if a.stage=='validate' else 'SUBMISSION.json')
    assert not record_path.exists(),'Already submitted'
    source=json.loads((r/'SOURCE_MANIFEST.json').read_text())
    cmd=['sbatch','--parsable','--partition=suma_rtx4090,asus_6000ada,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000','--qos=big_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40']
    env=dict(os.environ,STUDY_ROOT=str(r))
    if a.stage=='validate':
        cmd+=['--time=00:30:00','--job-name=reverse-L1M-check','--output='+str(rt/'slurm/validation_%j.out'),'--error='+str(rt/'slurm/validation_%j.err')]
        env['VALIDATION_ONLY']='1'
    else:
        assert json.loads((rt/'GPU_VALIDATION.json').read_text())['passed']
        cmd+=['--array=0-3%4','--output='+str(rt/'slurm/%A_%a.out'),'--error='+str(rt/'slurm/%A_%a.err')]
        env['VALIDATION_ONLY']='0'
    cmd+=[str(r/'source/experiments/kl_reverse_wide_1d/job.sbatch')]
    job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
    record=dict(time=time.time(),stage=a.stage,job=job,source_commit=source['commit'],command=cmd)
    record_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
