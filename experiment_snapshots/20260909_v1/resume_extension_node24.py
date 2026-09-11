from pathlib import Path
import json,subprocess,shutil,time

p=Path('/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1')
record_path=p/'extension_resume2_jobs.json'
if record_path.exists():
    print(record_path.read_text());raise SystemExit(0)
old_array='2193269';old_gate='2193270'
running=subprocess.check_output(['squeue','--noheader','--jobs='+old_array,'--states=RUNNING','--format=%i'],text=True).strip()
assert not running,('Wait for active tasks before replacing array',running)
tasks=json.loads((p/'extension_tasks.json').read_text());done=[];remaining=[]
for i,commands in enumerate(tasks):
    if all((Path(c[c.index('--out')+1])/'COMPLETE').exists() for c in commands):done.append(i)
    else:
        remaining.append(i)
        for c in commands:
            d=Path(c[c.index('--out')+1]);assert not d.exists() or (d/'COMPLETE').exists(),('Partial output needs targeted recovery',d)
assert done and remaining
archive=p/'extension_recovery_20260910_node24';archive.mkdir(exist_ok=True)
for name in ['extension_resume_jobs.json','extension_gate.sh']:
    dst=archive/name
    if not dst.exists():shutil.copy2(p/name,dst)
for root in [p/'code/analysis_boltzmann',Path('/lustre/hobbit9882/OptiQ/analysis_boltzmann')]:
    for name in ['campaign.py','repair_dispatch.py']:
        f=root/name
        if not f.exists():continue
        tag=('snapshot_' if root==p/'code/analysis_boltzmann' else 'repo_')+name
        if not (archive/tag).exists():shutil.copy2(f,archive/tag)
        f.write_text(f.read_text().replace('node02,node04,node05,node14,cs-gpu-01,node35','node02,node04,node05,node14,node24,cs-gpu-01,node35'))
subprocess.run(['scancel',old_gate],check=True)
subprocess.run(['scancel',old_array],check=True)
# Keep the original task indices and preserve all completed results.
spec=f'{remaining[0]}-{remaining[-1]}' if remaining==list(range(remaining[0],remaining[-1]+1)) else ','.join(map(str,remaining))
cmd=['sbatch','--parsable','--partition=base_suma_rtx3090,dell_rtx3090','--exclude=node02,node04,node05,node14,node24,cs-gpu-01,node35','--array='+spec+'%2','--time=06:00:00','--job-name=optiq-extension-resume','--output',str(p/'logs/%A_%a.log'),'--export',f'ALL,OPTIQ_ANALYSIS_ROOT={p}/code',str(p/'code/analysis_boltzmann/job.sh'),'analysis_boltzmann.campaign','--worker','--campaign',str(p),'--stage','extension']
array=subprocess.check_output(cmd,text=True).strip().split(';')[0]
record={'array':array,'gate':None,'previous_array':old_array,'previous_gate':old_gate,'completed_preserved':len(done),'tasks':len(remaining),'remaining_indices':remaining,'excluded_new_node':'node24','reason':'CUDA initialization failure; guard held pending tasks','recovery':str(archive),'time':time.time(),'command':cmd}
record_path.write_text(json.dumps(record,indent=2))
gate=subprocess.check_output(['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G','--time=02:00:00','--dependency=afterok:'+array,'--job-name=optiq-extension-gate','--output',str(p/'logs/gate-%j.log'),str(p/'extension_gate.sh')],text=True).strip().split(';')[0]
record['gate']=gate;record_path.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k not in ['command','remaining_indices']},indent=2))
