"""Read-only campaign heartbeat; keep snapshots without changing remote jobs."""
import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent/'monitoring'
REMOTE=r'''
import json,os,sys,time
from pathlib import Path
import numpy as np
task=sys.argv[1]
r=Path('/home/heechan/optiq-experiments/trg-dacer-priority-20260921')
read=lambda p:json.loads(p.read_text())
out={k:read(r/(k+'-'+task+'.json')) if (r/(k+'-'+task+'.json')).exists() else None
     for k in ('status','failure','selection','result','manifest','beta-cancellation')}
m=out['manifest']
assert m['controller_commit']=='1aa12756bcff953adbac8b4990f14b483e40651f'
assert m['beta_queue_canceled'] and out['status']['queued']==[]
assert len(out['beta-cancellation']['canceled'])==(7 if task=='humanoid' else 10)
assert all('dacer' in n for n in out['status']['running'])
assert read(r/'jobs'/(task+'-trg-dacer-T0.25-b1-s4.json'))['status']=='canceled'
out['runs']=[]
for seed in range(4):
    name=task+'-trg-dacer-T0.25-b1-s'+str(seed)
    j=read(r/'jobs'/(name+'.json'))
    dirs=list((r/'outputs').glob(name+'_*'));assert len(dirs)==1
    d=dirs[0];c=read(d/'config.json');a=c['alg']['actor']
    assert c['runtime']['git_commit']=='efe67fde083513b89f57a34a54743cb1e6d4b49d'
    assert a['temperature']==.25 and a['density_beta']==1 and c['dacer']['enabled']
    assert c['dacer']['target_entropy_per_dim']==-.9
    assert a['hidden_dims']==[256,256] and c['alg']['critic']['hs']==[256,256]
    assert a['log_std_min']==-5 and a['log_std_max']==-1 and a['initial_log_std']==-1
    assert c['alg']['batch_size']==256 and c['alg']['utd']==1
    assert c['alg']['optimizer']['lr_actor']==c['alg']['optimizer']['lr_critic']==.0003
    assert a['num_policy_samples']==64 and a['proposals_per_policy_sample']==1
    evals={}
    for mode in ('stochastic_z','zero_z'):
        files=list(d.rglob('evaluations_'+mode+'.npz'));assert len(files)==1
        with np.load(files[0],allow_pickle=False) as z:
            assert np.isfinite(z['results']).all()
            assert z['results'].shape==(len(z['timesteps']),10)
            evals[mode]=dict(step=int(z['timesteps'][-1]),reward=float(z['results'][-1].mean()),count=len(z['timesteps']))
    log=r/'logs'/(name+'.log')
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-16000));tail=f.read().decode(errors='replace')
    alive=False
    if j['status']=='running':
        os.kill(j['pid'],0)
        state=Path('/proc/'+str(j['pid'])+'/stat').read_text().rsplit(')',1)[1].split()[0]
        alive=state!='Z'
    if j['status']=='completed': assert all(v['step']==1000000 for v in evals.values())
    reg=d/'dacer_regulator.json'
    out['runs'].append(dict(seed=seed,pid=j.get('pid'),status=j['status'],alive=alive,
        evaluations=evals,regulator=read(reg) if reg.exists() else None,
        log_age_seconds=time.time()-log.stat().st_mtime,
        traceback_in_tail='Traceback (most recent call last)' in tail,
        error=j.get('error'),logging_warning=j.get('logging_warning')))
out['controller_tail']=(r/('controller-'+task+'.log')).read_text().splitlines()[-12:]
out['checked_at']=time.time()
print(json.dumps(out))
'''

def fetch(pair):
    host,task=pair
    p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=10',
        '-o','ServerAliveCountMax=2',host,'/home/heechan/.venv-optiq-mujoco/bin/python - '+task],
        input=REMOTE,text=True,capture_output=True,timeout=90)
    if p.returncode: raise RuntimeError(host+' '+p.stderr+' '+p.stdout)
    data=json.loads(p.stdout)
    ROOT.mkdir(exist_ok=True)
    prior=ROOT/(task+'-latest.json')
    old=json.loads(prior.read_text()) if prior.exists() else None
    text=json.dumps(data,indent=2)+'\n'
    prior.write_text(text)
    (ROOT/(task+'-'+str(int(data['checked_at']))+'.json')).write_text(text)
    runs=[]
    for run in data['runs']:
        prev=next((x for x in (old or {}).get('runs',[]) if x['seed']==run['seed']),None)
        step=run['evaluations']['stochastic_z']['step']
        delta=step-prev['evaluations']['stochastic_z']['step'] if prev else None
        runs.append(dict(seed=run['seed'],status=run['status'],step=step,progress_since_check=delta,
            alive=run['alive'],log_age_seconds=round(run['log_age_seconds'],1),
            traceback=run['traceback_in_tail'],error=run['error'],logging_warning=run['logging_warning']))
    return dict(task=task,failure=data['failure'],result_exists=data['result'] is not None,
        queue=data['status']['queued'],runs=runs)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for result in pool.map(fetch,[('vast-heechan-180','humanoid'),('vast-heechan-199','ant')]):
            print(json.dumps(result,indent=2))
