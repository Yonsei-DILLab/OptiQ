import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();(root/'runtime/slurm').mkdir(parents=True,exist_ok=True)
    record=root/'SUBMISSION.json';assert not record.exists()
    cmd=['sbatch','--parsable',
         '--partition=suma_rtx4090,asus_6000ada,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000',
         '--qos=big_qos','--array=0-14%8',
         '--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40',
         '--output='+str(root/'runtime/slurm/%A_%a.out'),
         '--error='+str(root/'runtime/slurm/%A_%a.err'),
         str(root/'source/experiments/kl_appendix_six/job.sbatch')]
    env=dict(os.environ,STUDY_ROOT=str(root),PARENT_ROOT=str(a.parent.resolve()))
    job=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
    payload=dict(job=job,indices='0-14%8',source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],command=cmd,time=time.time())
    record.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))

if __name__=='__main__':main()
