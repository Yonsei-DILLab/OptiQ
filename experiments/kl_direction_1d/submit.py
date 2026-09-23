"""Submit one correctness gate and 20 independent production tasks on login4."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    if (runtime/'SUBMISSION.json').exists():raise RuntimeError('Already submitted; do not duplicate jobs')
    logs=runtime/'slurm';logs.mkdir(exist_ok=True)
    partitions='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'
    common=['sbatch','--parsable','--partition='+partitions,'--qos=big_qos',
        '--exclude=cs-gpu-01,node14,node23,node24,node31,node35,node40',
        '--output='+str(logs/'%A_%a.out'),'--error='+str(logs/'%A_%a.err')]
    script=root/'source/experiments/kl_direction_1d/job.sbatch'
    env=dict(os.environ,STUDY_ROOT=str(root))
    validation=subprocess.check_output(common+['--time=00:40:00','--job-name=kl1d-check',str(script)],env=dict(env,VALIDATE_ONLY='1'),text=True).strip().split(';')[0]
    # The dependency is only a numerical correctness gate, never a method order.
    tasks=subprocess.check_output(common+['--array=0-19%8','--dependency=afterok:'+validation,str(script)],env=dict(env,VALIDATE_ONLY='0'),text=True).strip().split(';')[0]
    record=dict(time=time.time(),validation_job=validation,array_job=tasks,array='0-19%8',
        source_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],partitions=partitions)
    (runtime/'SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
