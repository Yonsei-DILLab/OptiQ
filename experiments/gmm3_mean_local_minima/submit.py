import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--validate',action='store_true');a=p.parse_args()
    root=a.root.resolve();(root/'slurm').mkdir(exist_ok=True)
    tag='validation' if a.validate else 'production';record=root/f'{tag}_SUBMISSION.json';assert not record.exists()
    if not a.validate:assert json.loads((root/'VALIDATION.json').read_text())['passed']
    cmd=['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos',
         '--output='+str(root/'slurm'/f'{tag}_%A_%a.out'),'--error='+str(root/'slurm'/f'{tag}_%A_%a.err')]
    cmd+=['--time=00:20:00'] if a.validate else ['--array=0-8%9']
    cmd+=[str(root/'source/experiments/gmm3_mean_local_minima/job.sbatch')]
    env=dict(os.environ,STUDY_ROOT=str(root),VALIDATION_ONLY='1' if a.validate else '0')
    job=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
    data=dict(job=job,tag=tag,time=time.time(),commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],command=cmd)
    record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
