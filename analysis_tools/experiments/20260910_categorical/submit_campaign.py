"""Submit only after the isolated categorical implementation passes validation."""
import argparse,json,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);a=ap.parse_args();p=Path(a.campaign)
smoke=json.loads((p/'smoke_passed.json').read_text())
assert smoke['passed'] is True
assert len(smoke['rows'])==3 and all(r.get('wrapper_matches_categorical') and r.get('wrapper_differs_from_argmax') for r in smoke['rows']), 'Actual frozen-Q categorical dispatch was not validated'
assert not (p/'jobs.json').exists(), 'Campaign already submitted; review scheduler state before resubmission'
tasks=json.loads((p/'tasks.json').read_text());assert len(tasks)==60
assert set(t['seed'] for t in tasks)==set(range(5))
code=p/'code'
cmd=['sbatch','--parsable','--partition=base_suma_rtx3090,dell_rtx3090',
     '--exclude=node02,node04,node05,node14,node24,cs-gpu-01,node35','--array=0-59%2',
     '--time=01:00:00','--job-name=optiq-categorical','--output='+str(p/'logs/%A_%a.log'),
     '--export=ALL,OPTIQ_ANALYSIS_ROOT='+str(code),str(code/'analysis_boltzmann/job.sh'),
     'analysis_boltzmann.categorical_worker','--campaign',str(p)]
array=subprocess.check_output(cmd,text=True).strip().split(';')[0]
record=dict(array=array,summary_job=None,tasks=60,seeds=list(range(5)),max_concurrent_gpus=2,command=cmd)
(p/'jobs.json').write_text(json.dumps(record,indent=2))
script=p/'summarize.sh'
script.write_text('''#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
cd '''+str(code)+'''
export PYTHONPATH="$PWD"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m analysis_boltzmann.categorical_summary --campaign '''+str(p)+''' --require-complete
''')
cmd=['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G',
     '--time=00:30:00','--dependency=afterany:'+array,'--job-name=optiq-cat-summary',
     '--output='+str(p/'logs/summary-%j.log'),str(script)]
summary=subprocess.check_output(cmd,text=True).strip().split(';')[0];record.update(summary_job=summary,summary_command=cmd)
(p/'jobs.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
