"""Submit after both numerical-reference and actual GPU update validation."""
import argparse,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
args=parser.parse_args();root=Path(args.campaign)
smoke=json.loads((root/'smoke_passed.json').read_text())
assert smoke['passed'] and smoke['paired_initialization_verified']
assert len(smoke['rows'])==6 and all(row['loss_verified'] for row in smoke['rows'])
reference=json.loads((root/'references/passed.json').read_text());assert reference['passed']
assert len(reference['cases'])==14
assert not (root/'jobs.json').exists(),'Already submitted'
tasks=json.loads((root/'tasks.json').read_text());assert len(tasks)==140
command=['sbatch','--parsable','--partition=base_suma_rtx3090,dell_rtx3090',
    '--exclude=node02,node04,node05,node14,node24,cs-gpu-01,node35',
    '--array=0-139%2','--time=06:00:00','--job-name=optiq-batch1',
    '--output='+str(root/'logs/%A_%a.log'),'--export=ALL,OPTIQ_ANALYSIS_ROOT='+str(root/'code'),
    str(root/'code/analysis_boltzmann/job.sh'),'analysis_batch1.worker','--campaign',str(root)]
array=subprocess.check_output(command,text=True).strip().split(';')[0]
record=dict(array=array,tasks=len(tasks),max_concurrent_gpus=2,command=command)
(root/'jobs.json').write_text(json.dumps(record,indent=2))
script=root/'summarize.sh'
script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'+
    'exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python '+str(root/'summarize.py')+
    ' --campaign '+str(root)+' --require-complete\n')
command=['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G',
    '--time=00:30:00','--dependency=afterany:'+array,'--job-name=optiq-batch1-summary',
    '--output='+str(root/'logs/summary-%j.log'),str(script)]
record['summary_job']=subprocess.check_output(command,text=True).strip().split(';')[0]
record['summary_command']=command
(root/'jobs.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
