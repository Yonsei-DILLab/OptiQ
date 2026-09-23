import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();rt=r/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    assert not (rt/'SUBMISSION.json').exists(),'Already submitted'
    source=json.loads((r/'SOURCE_MANIFEST.json').read_text())
    cmd=['sbatch','--parsable','--partition=big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000','--qos=big_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40','--array=0-3%4','--output='+str(rt/'slurm/%A_%a.out'),'--error='+str(rt/'slurm/%A_%a.err'),str(r/'source/experiments/kl_forward_width05_1d/job.sbatch')]
    job=subprocess.check_output(cmd,env=dict(os.environ,STUDY_ROOT=str(r)),text=True).strip().split(';')[0]
    record=dict(time=time.time(),job=job,source_commit=source['commit'],command=cmd)
    (rt/'SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
