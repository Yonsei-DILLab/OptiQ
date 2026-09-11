"""Predeclared group budget gates for four manifested proximal Humanoid runs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.monitor_v2_finite import evaluations,strong_group_decision
BASE=ROOT/'outputs/v2_improvement'


def checkpoint_records(runs,step):
    records=[]
    for run in runs:
        for kind in ('actor','critic'):
            files=list(Path(run['directory']).glob(f'checkpoints/*/{kind}_state_{step}.msgpack'))
            if len(files)!=1 or files[0].stat().st_size==0:
                return None
            records.append({'seed':run['seed'],'kind':kind,'path':str(files[0]),
                            'sha256':hashlib.sha256(files[0].read_bytes()).hexdigest()})
    return records


def service_status(service):
    output=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True).stdout.split()
    return output[1] if len(output)>1 else 'UNKNOWN'


def main():
    manifest=json.loads((BASE/'proximal_screen_manifest.json').read_text())
    protocol=json.loads((BASE/'proximal_screen_protocol.json').read_text())
    runs=manifest['runs']
    assert sorted(r['seed'] for r in runs)==list(range(4))
    for r in runs: assert r['service']==f'optiq-v2-proximal:optiq-v2-proximal-{r["seed"]}'
    refs=json.loads((BASE/'confirmation_manifest.json').read_text())['runs']
    continuous={r['seed']:evaluations(r['directory']) for r in refs if r['method']=='v2'}
    historical={}
    for r in json.loads((BASE/'historical_behavior010_reference.json').read_text())['runs']:
        path=next(Path(r['directory']).glob('eval/*/evaluations.npz'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['evaluation_sha256']
        historical[r['seed']]=evaluations(r['directory'])
    path=BASE/'proximal_screen_monitor.json'
    state=json.loads(path.read_text()) if path.exists() else {
        'started_utc':datetime.now(timezone.utc).isoformat(),'reviews':[],'stops':[],
        'stage':'monitoring','goal_complete':False}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
    while True:
        state['statuses']={str(r['seed']):service_status(r['service']) for r in runs}
        # An incomplete terminal run remains visible, never dropped or restarted.
        terminal={'EXITED','STOPPED','FATAL'}
        if all(s in terminal for s in state['statuses'].values()):
            complete=all((Path(r['directory'])/'completed.json').exists() and
                         json.loads((Path(r['directory'])/'completed.json').read_text()).get('timesteps')==1000000
                         for r in runs)
            state['stage']='training_finished_requires_final_evaluation' if complete else 'screen_incomplete'
            save();return
        try:
            curves={r['seed']:evaluations(r['directory']) for r in runs}
            state['latest_evaluation_steps']={str(s):max(v) for s,v in curves.items()}
            for gate in protocol['budget_gates']:
                if any(r['step']==gate['step'] and r.get('applied') for r in state['reviews']):continue
                review=strong_group_decision(curves,continuous,historical,gate['step'],gate['threshold'])
                if not review['available']:break
                review['reason']='predeclared_proximal_budget_gate'
                if review['stop']:
                    checkpoints=checkpoint_records(runs,gate['step'])
                    if checkpoints is None:break
                    review.update(checkpoints=checkpoints,applied=False)
                    state['reviews'].append(review);save()
                    for r in runs:
                        if service_status(r['service']) in terminal:continue
                        result=subprocess.run(['supervisorctl','stop',r['service']],capture_output=True,text=True)
                        state['stops'].append({'seed':r['seed'],'step':gate['step'],'returncode':result.returncode,
                            'supervisor_response':result.stdout.strip()});save()
                    review['applied']=all(service_status(r['service']) in terminal for r in runs)
                    save();break
                review['applied']=True;state['reviews'].append(review)
        except (OSError,ValueError,EOFError,StopIteration) as error:
            state['last_read_error']=str(error)
        save();time.sleep(30)


if __name__=='__main__':main()
