import argparse,json,os,subprocess,time
from pathlib import Path
POOLS={
 'ordinary':('suma_rtx4090,asus_6000ada,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000','big_qos'),
 'pro6000':('asus_pro6000,gigabyte_pro6000','pro6000_qos'),
 'a100':('suma_a100','a100_qos')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pool',choices=list(POOLS),default='ordinary');p.add_argument('--stage',choices=['validate','confirm'],required=True);p.add_argument('--indices');p.add_argument('--tag',default='first');a=p.parse_args()
 root=a.root.resolve();rt=root/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True);cfg=json.loads((root/'source/experiments/kl_six_highL_1d/config.json').read_text());assert cfg['allow_large_L'] and cfg['reverse_L']==1048576
 record=rt/f'{a.stage}_{a.pool}_{a.tag}_SUBMISSION.json';assert not record.exists()
 if a.stage=='confirm':
  assert a.indices and json.loads((rt/f'GPU_VALIDATION_{a.pool}.json').read_text())['passed']
  assert json.loads((root/'APPROVAL.json').read_text())['approved']
 partition,qos=POOLS[a.pool];cmd=['sbatch','--parsable',f'--partition={partition}',f'--qos={qos}',
  '--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40',
  '--output='+str(rt/'slurm'/f'{a.stage}_{a.pool}_{a.tag}_%A_%a.out'),
  '--error='+str(rt/'slurm'/f'{a.stage}_{a.pool}_{a.tag}_%A_%a.err')]
 if a.stage=='validate':cmd+=['--time=00:30:00']
 else:cmd+=['--array='+a.indices]
 cmd+=[str(root/'source/experiments/kl_six_highL_1d/job.sbatch')]
 env=dict(os.environ,STUDY_ROOT=str(root),POOL=a.pool,VALIDATION_ONLY='1' if a.stage=='validate' else '0')
 job=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
 data=dict(job=job,stage=a.stage,pool=a.pool,indices=a.indices,time=time.time(),source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],command=cmd)
 record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
