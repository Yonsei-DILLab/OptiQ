"""Resume only unfinished commands; preserve all complete runs and failed artifacts."""
from pathlib import Path
import hashlib,json,subprocess,time,shutil
p=Path('/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1')
root=p.parents[2]; code=p/'code'; stamp=time.strftime('%Y%m%d_%H%M%S'); audit=p/('recovery_'+stamp); audit.mkdir()
source=root/'analysis_boltzmann'; patchfiles=['job.sh','campaign.py','learned_backup.py','quadrature_check.py']
changes={}
for name in patchfiles:
    target=code/'analysis_boltzmann'/name
    old=target.read_bytes() if target.exists() else b''
    (audit/(name+'.before')).write_bytes(old)
    new=(source/name).read_bytes(); changes[name]={'before':hashlib.sha256(old).hexdigest(),'after':hashlib.sha256(new).hexdigest()}
    target.write_bytes(new)
tasks=json.loads((p/'pilot_tasks.json').read_text()); unfinished=[]; archived=[]
for index,commands in enumerate(tasks):
    if all((Path(c[c.index('--out')+1])/'COMPLETE').exists() for c in commands): continue
    unfinished.append(index)
    for c in commands:
        out=Path(c[c.index('--out')+1]); assert out.parent==p/'runs'
        if out.exists() and not (out/'COMPLETE').exists():
            destination=audit/out.name; out.rename(destination); archived.append({'original':str(out),'archived':str(destination)})
assert len(unfinished)==108,('Unexpected current state; inspect before submitting',len(unfinished))
(audit/'manifest.json').write_text(json.dumps({'unfinished_indices':unfinished,'archived':archived,'patches':changes,'training_code_changed':False,'change':'Exclude node04, hold pending jobs on GPU preflight failure, higher-precision converged Q diagnostics'},indent=2))
subprocess.run(['scancel','2179738'],check=True)
# Require successful full-checkpoint quadrature validation before the array starts.
cmd=['sbatch','--parsable','--partition=base_suma_rtx3090','--exclude=node02,node04,node14,node35,cs-gpu-01','--dependency=afterok:2182278','--array',','.join(map(str,unfinished))+'%2','--job-name','optiq-pilot-resume','--output',str(p/'logs/%A_%a.log'),'--export','ALL,OPTIQ_ANALYSIS_ROOT='+str(code),str(code/'analysis_boltzmann/job.sh'),'analysis_boltzmann.campaign','--worker','--campaign',str(p),'--stage','pilot']
job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
record={'array':job,'gate':None,'tasks':108,'previous_array':'2179729','recovery':str(audit),'command':cmd}
(p/'pilot_resume_jobs.json').write_text(json.dumps(record,indent=2))
gate=subprocess.check_output(['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G','--time=02:00:00','--dependency=afterok:'+job,'--job-name','optiq-gate-resume','--output',str(p/'logs/gate-%j.log'),str(p/'pilot_gate.sh')],text=True).strip().split(';')[0]
record['gate']=gate;(p/'pilot_resume_jobs.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
