"""Submit only L1024 screening/held-out seeds. Never submit high-L confirmation."""
import argparse,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['validate','screen','validate_seeds'],required=True)
    p.add_argument('--indices');p.add_argument('--tag',default='first');p.add_argument('--config',default='config.json',choices=['config.json','shape_config.json']);args=p.parse_args()
    root=args.root.resolve();rt=root/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'source/experiments/kl_diverse_targets_1d'/args.config).read_text())
    assert not cfg['allow_large_L'] and cfg['reverse_L']==1024
    if args.stage!='validate':assert json.loads((rt/'GPU_VALIDATION.json').read_text())['passed']
    name=f'{args.stage}_{args.tag}';record=rt/f'{name}_SUBMISSION.json';assert not record.exists()
    cmd=['sbatch','--parsable','--partition=suma_rtx4090,asus_6000ada,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000',
         '--qos=base_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40',
         '--output='+str(rt/'slurm'/f'{name}_%A_%a.out'),'--error='+str(rt/'slurm'/f'{name}_%A_%a.err')]
    env=dict(os.environ,STUDY_ROOT=str(root),STAGE=args.stage,CONFIG_NAME=args.config,VALIDATION_ONLY='1' if args.stage=='validate' else '0')
    if args.stage=='validate':cmd+=['--time=00:20:00']
    else:cmd+=['--array='+ (args.indices or f'0-{len(cfg["cases"])*len(cfg["seeds"])-1}%8')]
    cmd+=[str(root/'source/experiments/kl_diverse_targets_1d/job.sbatch')]
    job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
    data=dict(job=job,time=time.time(),stage=args.stage,config=args.config,command=cmd,source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'])
    record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))

if __name__=='__main__':main()
