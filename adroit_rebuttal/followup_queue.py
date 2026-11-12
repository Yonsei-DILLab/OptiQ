"""Durable dependency-gated DIPO -> MEOW -> MFPO continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .run import atomic, guard


def upstream_ready(path, program):
    if not path.exists():
        return False, 'upstream plan absent'
    d=json.loads(path.read_text())
    jobs=d.get('jobs',[])
    if len(jobs)!=40 or {(j['method'],j['task'],j['seed']) for j in jobs} != {
        (m,t,s) for m in ('ibolt','sac') for t in ('door','hammer','pen','relocate') for s in range(5)}:
        return False,'upstream cohort identity mismatch'
    if any(j['status']=='failed' for j in jobs):
        return False,'upstream has failed runs; waiting for review'
    if d['status']!='completed' or not all(j['status']=='completed' for j in jobs):
        return False,'waiting for all 40 iBOLT/SAC runs'
    for j in jobs:
        p=Path(j['output'])/'completed.json'
        if not p.exists():return False,'upstream completion marker absent'
        r=json.loads(p.read_text())
        if not (r.get('completed') and not r.get('smoke') and r['steps']==5000192 and r['updates']==623424 and
                r['num_envs']==256 and r['observed_utd']==.125 and
                (r['method'],r['task'],r['seed'])==(j['method'],j['task'],j['seed'])):
            return False,'upstream completion marker mismatch'
        if j['task'] in ('hammer','pen'):
            proof_path=Path(j['output'])/'demonstration-validation.json'
            if not proof_path.exists():return False,'verified demonstration proof absent'
            proof=json.loads(proof_path.read_text())
            if not (proof.get('passed') and proof.get('replayed') and proof['trajectories']==25 and
                    proof['manifest_sha256']==j.get('demonstration_manifest_sha256')):
                return False,'verified demonstration proof mismatch'
    status=subprocess.run(['sudo','-n','supervisorctl','status',program],text=True,stdout=subprocess.PIPE).stdout.split()
    if len(status)<2 or status[1] in ('RUNNING','STARTING','STOPPING','BACKOFF'):
        return False,'waiting for upstream supervisor program to exit'
    return True,'all upstream runs verified'


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    guard();cfg=json.loads(a.config.read_text());root=a.config.parent
    def state(status,**fields):
        atomic(root/'queue-state.json',dict(status=status,observed_unix=time.time(),**fields))
    last=None
    while True:
        ready,reason=upstream_ready(Path(cfg['upstream_plan']),cfg['upstream_program'])
        compat=Path(cfg['compatibility'])
        checked=compat.exists() and json.loads(compat.read_text()).get('passed',False)
        state('waiting',upstream_ready=ready,compatibility_ready=checked,reason=reason)
        if reason!=last:print(reason,flush=True);last=reason
        if ready and checked:break
        time.sleep(30)
    manifest=json.loads(Path('/workspace/heechan-adroit/datasets/verified-demonstrations.json').read_text())
    assert manifest['passed']
    for d in manifest['datasets']:
        if d['task'] in ('hammer','pen'):
            from .verified_demos import resolve
            resolve(d['task'],cfg['verified_demonstrations'][d['task']])
            continue
        data=Path('/workspace/heechan-adroit/datasets')/(d['task']+'-human-sparse-v2.npz')
        assert hashlib.sha256(data.read_bytes()).hexdigest()==d['converted_sha256']
    for method in ('dipo','meow','mfpo'):
        for phase in ('smoke','full'):
            plan=Path(cfg['phases'][method][phase]);d=json.loads(plan.read_text())
            if d['status']=='completed' and all(j['status']=='completed' for j in d['jobs']):continue
            if d['status']!='prepared':
                state('held',method=method,phase=phase,reason='Phase requires review; no automatic replay resume or retry')
                return
            state('running',method=method,phase=phase,plan=str(plan))
            print('Starting '+method+' '+phase,flush=True)
            subprocess.run([cfg['python'],'-u','-m','adroit_rebuttal.baseline_campaign','--plan',str(plan)],
                           cwd=cfg['source'],check=True)
            d=json.loads(plan.read_text())
            if d['status']!='completed' or not all(j['status']=='completed' for j in d['jobs']):
                state('held',method=method,phase=phase,reason='Phase has failures; later methods have not been released')
                return
    state('completed',reason='All 60 full runs and 12 vectorized smoke checks completed')


if __name__=='__main__':main()
