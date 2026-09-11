"""Preserve original control runs and dispatch the validated numerical revision."""
import argparse,json,shutil,subprocess,time,hashlib
from pathlib import Path


def dispatch(campaign):
    p=Path(campaign).resolve();rec=p/'control_repair_20260909';code=p/'code';jf=rec/'jobs.json'
    if jf.exists():
        print(jf.read_text(),flush=True);return
    assert (rec/'smoke_cpu/COMPLETE').exists(),'Control smoke not passed'
    audit=p/'control_precision_audit/all256'
    assert (audit/'COMPLETE').exists(),'Original-grid audit not complete'
    diagnostics=json.loads((audit/'summary.json').read_text());assert len(diagnostics)==20
    assert all(r['converged'] for r in diagnostics),'Original-critic reference needs further refinement'
    originals=[p/'runs'/f'control_{arm}_seed{seed}' for seed in range(5) for arm in ['max','boltzmann','actor','local']]
    archived=rec/'original_runs';archived.mkdir(exist_ok=True)
    for d in originals:
        assert (d/'COMPLETE').exists() or (archived/d.name/'COMPLETE').exists(),d
    if not (rec/'original_pilot_tasks.json').exists():shutil.copy2(p/'pilot_tasks.json',rec/'original_pilot_tasks.json')
    tasks=json.loads((rec/'original_pilot_tasks.json').read_text());repair=[]
    for commands in tasks:
        for cmd in commands:
            if cmd[0]!='analysis_boltzmann.control':continue
            if '--grid' in cmd:cmd[cmd.index('--grid')+1]='2056'
            else:cmd+=['--grid','2056']
            repair.append([cmd])
    assert len(repair)==20
    for d in originals:
        if d.exists():
            assert not (archived/d.name).exists(),('Archive collision',d)
            d.rename(archived/d.name)
    (p/'pilot_tasks.json').write_text(json.dumps(tasks,indent=2));(p/'control_repair_tasks.json').write_text(json.dumps(repair,indent=2))
    protocol=json.loads((p/'protocol.json').read_text());protocol['control_numeric_revision']=json.loads((rec/'plan.json').read_text());(p/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cmd=['sbatch','--parsable','--partition=base_suma_rtx3090,dell_rtx3090','--exclude=node02,node04,node05,node14,node24,cs-gpu-01,node35','--array=0-19%2','--time=02:00:00','--job-name=optiq-control-repair','--output',str(p/'logs/%A_%a.log'),'--export',f'ALL,OPTIQ_ANALYSIS_ROOT={code}',str(code/'analysis_boltzmann/job.sh'),'analysis_boltzmann.campaign','--worker','--campaign',str(p),'--stage','control_repair']
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    record={'array':jid,'gate':None,'command':cmd,'tasks':20,'time':time.time(),'preserved_originals':str(archived),'smoke':str(rec/'smoke_cpu/summary.json')};jf.write_text(json.dumps(record,indent=2))
    # Re-run the original full-pilot validation, then use its crosspilot -> extension chain.
    gate=subprocess.check_output(['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G','--time=02:00:00','--dependency=afterok:'+jid,'--job-name=optiq-repair-gate','--output',str(p/'logs/gate-%j.log'),str(p/'pilot_gate.sh')],text=True).strip().split(';')[0]
    record['gate']=gate;jf.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);a=ap.parse_args();dispatch(a.campaign)
