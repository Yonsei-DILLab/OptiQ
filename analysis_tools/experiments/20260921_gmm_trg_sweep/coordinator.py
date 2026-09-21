"""Run on the Mac: bounded global stage transitions, no SSH keys copied to servers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from campaign import job, TEMPS, SLOTS

ROOT='/workspace/optiq-gmm-trg-20260921'
SCRIPT=ROOT+'/code/analysis_tools/experiments/20260921_gmm_trg_sweep/campaign.py'
LOCAL=Path('/Users/jaehun/Documents/online-RL/experiments/gmm_trg_20260921')
PY={h:'/workspace/optiq-v5-comparison-20260913/venv/bin/python' for h in SLOTS}
PY['vast5']='/workspace/optiq-ant-highbeta-20260916/venv/bin/python'

def remote(host,command,data=None,timeout=90):
    cmd=f'sudo -n env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={ROOT}/deps {PY[host]} {SCRIPT} {command}'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,cmd],
                     input=None if data is None else json.dumps(data),text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'{host}: {p.stderr[-1000:]} {p.stdout[-1500:]}')
    lines=[line for line in p.stdout.splitlines() if line.startswith(('{','['))]
    return json.loads(lines[-1])

def phase_jobs(stage,best=None):
    if stage==1:
        return [job(e,t,s) for e,ts in TEMPS.items() for t in ts for s in range(5)]
    if stage==2:
        return [job(e,t,s,dacer=True,stage=2) for e,ts in TEMPS.items()
                if e not in ('ant','humanoid') for t in ts for s in range(5)]
    if stage==3:
        return [job(e,best[e],s,beta=b,stage=3) for e in TEMPS for b in (.5,.9) for s in range(5)]
    if stage==4:
        return [job(e,best[e],s,dacer=True,stage=4) for e in ('ant','humanoid') for s in range(5)]

def initial_allocation():
    plan={h:[] for h in SLOTS}
    def add(h,e,t,seeds):plan[h].extend(job(e,t,s) for s in seeds)
    add('vast5','humanoid',.25,[3,4]);add('vast5','ant',.25,[3,4])
    add('vast5','halfcheetah',.1,[0,1]);add('vast5','walker2d',.1,[0,1])
    add('vast2','halfcheetah',.1,[2,3,4])
    add('vast1','halfcheetah',.25,[3,4]);add('vast1','walker2d',.25,[3,4])
    add('vast1','walker2d',.1,[2,3])
    add('vast3','walker2d',.1,[4]);add('vast3','hopper',.05,range(5))
    add('vast4','hopper',.1,range(5))
    assert sum(map(len,plan.values()))==28
    for jobs in plan.values():
        for i,j in enumerate(jobs):j['order']=i
    return plan

def snapshots():
    with ThreadPoolExecutor(max_workers=5) as pool:
        result=dict(zip(SLOTS,pool.map(lambda h:remote(h,'snapshot'),SLOTS)))
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def index(states):
    lookup={}
    for host,x in states.items():
        for j in x['jobs']:
            assert j['id'] not in lookup, 'Duplicate scientific run: '+j['id']
            lookup[j['id']]=dict(j,host=host)
    return lookup

def finished(j):return j['status'] in ('completed','trained_logging_failed') and bool(j.get('scores'))

def best_temperatures(lookup):
    best={};report={}
    for env,temps in TEMPS.items():
        report[env]={}
        for t in temps:
            rows=[lookup[job(env,t,s)['id']] for s in range(5)]
            assert all(finished(j) for j in rows)
            values=[j['scores']['stochastic_z'] for j in rows]
            report[env][str(t)]={'seed_scores':values,'mean':sum(values)/5,
                                  'ids':[j['wandb_id'] for j in rows]}
        best[env]=min(temps,key=lambda t:(-report[env][str(t)]['mean'],t))
    return best,report

def distribute(jobs):
    # Greedy duration-balanced host allocations, prioritizing heavy environments.
    speed={'vast1':1.6,'vast2':2.0,'vast3':1.0,'vast4':1.0,'vast5':2.0}
    cost={'humanoid':1.7,'ant':1.2,'halfcheetah':1.,'walker2d':1.,'hopper':.8}
    loads={h:0. for h in SLOTS};plan={h:[] for h in SLOTS}
    for j in sorted(jobs,key=lambda j:(-cost[j['task']],j['id'])):
        h=min(SLOTS,key=lambda h:(loads[h]+cost[j['task']])/(speed[h]*len(SLOTS[h])))
        j['order']=len(plan[h]);plan[h].append(j);loads[h]+=cost[j['task']]
    return plan

def advance():
    states=snapshots();lookup=index(states)
    stages={x['gate']['stage'] for x in states.values()}
    stage=min(stages)
    # Recover a partially applied gate transition from its prewritten manifest.
    if len(stages)>1:
        next_stage=max(stages)
        assert next_stage==stage+1 and (LOCAL/f'stage{next_stage}.json').exists()
        manifest=json.loads((LOCAL/f'stage{next_stage}.json').read_text())
        for h,jobs in manifest.items():remote(h,'add',jobs)
        for h in SLOTS:remote(h,f'gate {next_stage}')
        return {'recovered_gate':next_stage}
    best=None
    if stage>=3:best,_=best_temperatures(lookup)
    expected=phase_jobs(stage,best)
    missing=[j['id'] for j in expected if j['id'] not in lookup or not finished(lookup[j['id']])]
    if missing:return {'stage':stage,'remaining':len(missing),'ids':missing}
    if stage==4:return {'complete':True,'unique_runs':len(lookup)}
    best,report=best_temperatures(lookup)
    (LOCAL/'selection.json').write_text(json.dumps({'best':best,'scores':report},indent=2)+'\n')
    future=phase_jobs(stage+1,best)
    manifest_path=LOCAL/f'stage{stage+1}.json'
    if manifest_path.exists(): plan=json.loads(manifest_path.read_text())
    else:
        plan=distribute(future);manifest_path.write_text(json.dumps(plan,indent=2)+'\n')
    assert {j['id'] for jobs in plan.values() for j in jobs}=={j['id'] for j in future}
    for host,jobs in plan.items():remote(host,'add',jobs)
    for host in SLOTS:remote(host,f'gate {stage+1}')
    return {'activated_stage':stage+1,'new_runs':len(future),'best':best}

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='plan':result=initial_allocation()
    elif mode=='status':result=snapshots()
    elif mode=='advance':result=advance()
    elif mode=='import':
        with ThreadPoolExecutor(max_workers=5) as pool:
            result=dict(zip(SLOTS,pool.map(lambda h:remote(h,'import-old',timeout=2400),SLOTS)))
    elif mode=='setup':
        plan=initial_allocation();LOCAL.mkdir(parents=True,exist_ok=True)
        (LOCAL/'stage1.json').write_text(json.dumps(plan,indent=2)+'\n')
        with ThreadPoolExecutor(max_workers=5) as pool:
            result=dict(zip(SLOTS,pool.map(lambda h:remote(h,'setup '+h,plan[h],timeout=120),SLOTS)))
    else:raise ValueError(mode)
    print(json.dumps(result,indent=2))
