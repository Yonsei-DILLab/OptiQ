"""Read-only W&B final summary check, with provenance-only sidecar."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
CODE=r'''
from pathlib import Path
from datetime import datetime,timezone
import json,math,wandb
root=Path('/home/heechan/optiq-experiments/antmaze-optiq-sigmacap-higher-v34-250k-s0-20260925')
source='b111a993b1e1bf895966dac4469abdfdb2c37c05'
read=lambda p:json.loads(p.read_text())
manifest=read(root/'manifest.json')
api=wandb.Api(timeout=60)
records=[]
for job in manifest['jobs']:
    folder=root/'runs'/job['id']
    if job['actor_sigma_profile']=='uncapped-initm1':
        assert not (folder/'result.json').exists()
        continue
    result=read(folder/'result.json')
    assert result['source_commit']==source and result['completed']
    info=read(folder/'wandb.json');run=api.run('OptiQ/antmaze/'+info['id'])
    s=run.summary_metrics
    checks={'finished':run.state=='finished','completed':s.get('completed') is True,
            'steps':s.get('steps')==result['steps']==258304,
            'updates':s.get('updates')==result['updates']==7816}
    for mode,stats in result['summaries'].items():
        for k in ['success_rate','episodes','mean_return']:
            key=f'final/{mode}/{k}';actual=s.get(key)
            checks[key]=actual is not None and math.isclose(actual,stats[k],rel_tol=0,abs_tol=1e-9)
    records.append(dict(id=job['id'],run_id=info['id'],url=info['url'],state=run.state,
        verified=all(checks.values()),checks=checks,
        summary={k:v for k,v in s.items() if k in ('completed','step','steps','updates') or k.startswith('final/')}))
record=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=source,
            verified=all(r['verified'] for r in records),records=records,
            script_sha256=SCRIPT_SHA,history_or_summary_changed=False,training_restarted=False)
(root/'wandb-final-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['vast-heechan-180','vast-heechan-199'],required=True)
    host=p.parse_args().host
    code='SCRIPT_SHA='+repr(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())+'\n'+CODE
    cmd='bash -c '+shlex.quote('source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm >/dev/null; exec /home/heechan/.venv-ddiffpg-native/bin/python -')
    p=subprocess.run(['ssh','-o','BatchMode=yes',host,cmd],input=code,text=True,capture_output=True,timeout=90)
    assert p.returncode==0,p.stderr[-1800:]
    d=json.loads(p.stdout.splitlines()[-1]);(ROOT/('wandb-final-verification-'+host+'.json')).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(verified=d['verified'],runs=[dict(run_id=r['run_id'],state=r['state'],verified=r['verified'],failed_checks=[k for k,v in r['checks'].items() if not v]) for r in d['records']])))
