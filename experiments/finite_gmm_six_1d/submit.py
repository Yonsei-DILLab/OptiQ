import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--validate',action='store_true');a=p.parse_args()
    root=a.root.resolve();(root/'slurm').mkdir(exist_ok=True)
    tag='validation' if a.validate else 'production';record=root/f'{tag}_SUBMISSION.json';assert not record.exists()
    if not a.validate:assert json.loads((root/'GPU_VALIDATION.json').read_text())['passed']
    cmd=['sbatch','--parsable','--partition=suma_rtx4090,asus_6000ada,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000','--qos=big_qos',
         '--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40',
         '--output='+str(root/'slurm'/f'{tag}_%A_%a.out'),'--error='+str(root/'slurm'/f'{tag}_%A_%a.err')]
    cmd+=['--time=00:20:00'] if a.validate else ['--array=0-74%12']
    cmd+=[str(root/'source/experiments/finite_gmm_six_1d/job.sbatch')]
    env=dict(os.environ,STUDY_ROOT=str(root),VALIDATION_ONLY='1' if a.validate else '0')
    job=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
    data=dict(job=job,tag=tag,time=time.time(),commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],command=cmd)
    record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
