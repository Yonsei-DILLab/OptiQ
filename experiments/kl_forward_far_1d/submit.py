import json,os,subprocess,time
from pathlib import Path
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();rt=root/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    if (rt/'SUBMISSION.json').exists():raise RuntimeError('Already submitted')
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for n in (64,128,256,512):
        d=root/f'N{n}M{n}';d.mkdir(exist_ok=True);(d/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cmd=['sbatch','--parsable','--partition=big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000','--qos=big_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40','--array=0-15%16','--output='+str(rt/'slurm/%A_%a.out'),'--error='+str(rt/'slurm/%A_%a.err'),str(root/'source/experiments/kl_forward_far_1d/job.sbatch')]
    job=subprocess.check_output(cmd,env=dict(os.environ,STUDY_ROOT=str(root)),text=True).strip().split(';')[0]
    result=dict(time=time.time(),job=job,source_commit=manifest['commit'],command=cmd);(rt/'SUBMISSION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
