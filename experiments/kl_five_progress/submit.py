"""Disjoint arrays, one GPU per trajectory; no duplicated competing jobs."""
import argparse,json,os,subprocess,time
from pathlib import Path
from ..kl_six_highL_1d.submit import POOLS
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pool',choices=POOLS,required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--indices');a=p.parse_args();root=a.root;rt=root/'runtime'
    (rt/'slurm').mkdir(parents=True,exist_ok=True)
    tag='preflight' if a.preflight else 'runs_'+a.indices.replace(',','_').replace('%','_')
    record=rt/f'{a.pool}_{tag}_SUBMISSION.json';assert not record.exists()
    if not a.preflight:
        assert a.indices and json.loads((rt/f'PREFLIGHT_{a.pool}.json').read_text())['passed']
    part,qos=POOLS[a.pool]
    cmd=['sbatch','--parsable',f'--partition={part}',f'--qos={qos}',
        '--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40',
        '--output='+str(rt/'slurm'/f'{a.pool}_{tag}_%A_%a.out'),
        '--error='+str(rt/'slurm'/f'{a.pool}_{tag}_%A_%a.err')]
    cmd+=['--time=00:30:00'] if a.preflight else ['--array='+a.indices]
    cmd+=[str(root/'source/experiments/kl_five_progress/job.sbatch')]
    env=dict(os.environ,STUDY_ROOT=str(root),POOL=a.pool,VALIDATION_ONLY='1' if a.preflight else '0')
    job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
    result=dict(job=job,pool=a.pool,indices=a.indices,preflight=a.preflight,time=time.time(),command=cmd,
        source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'])
    record.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
