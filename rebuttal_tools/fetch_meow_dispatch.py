"""Distribute existing MEOW trials; selection remains with one coordinator."""
import ast
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.dispatch.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read_peer(plan,peer):
    args=['ssh','-T','-n','-p',str(peer['port']),'-i',plan['dispatch_private_key'],
        '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8',
        '-o','IPQoS=none',
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+plan['dispatch_known_hosts'],
        peer['user']+'@'+peer['host'],'status']
    r=subprocess.run(args,capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr[-500:])
    data=json.loads(r.stdout)
    assert data['campaign']==plan['campaign'] and data['host']==peer['id']
    return data


def destination(plan,j):
    if j['phase']=='final':return 2 if j['seed'] in (0,1) else 3
    return plan['dispatch_assignments'][j['id']]


def mark_delegated(plan,j):
    host=destination(plan,j)
    j['delegation']=dict(host=host,job_id=j['id'],protocol='read-only remote result mirroring')
    for key in ('pid','start_ticks','gpu','learner_cpus','worker_cpus'):
        j.pop(key,None)


def mirror(plan,j,remote,host):
    assert remote['id']==j['id'] and remote['method']=='meow'
    assert all(remote[k]==j[k] for k in ('seed','task','phase','value'))
    assert remote['status'] in ('queued','initializing','running','completed','failed')
    out=Path(j['output']);out.mkdir(parents=True,exist_ok=True)
    origin=dict(host=host,output=remote['output'],job_id=remote['id'],mirrored_unix=time.time(),
        record_type='Remote metrics mirror; model and replay checkpoints remain on the execution server')
    atomic(out/'remote-origin.json',origin)
    j['status']=remote['status'];j['remote_output']=remote['output'];j['remote_checked_unix']=time.time()
    if remote.get('progress'):
        atomic(out/'progress.json',dict(remote['progress'],remote_origin=origin))
        j['progress']=remote['progress']
    for key,name in [('config','config.json'),('entropy_audit','entropy-parameter-audit.json')]:
        if remote.get(key):atomic(out/name,dict(remote[key],remote_origin=origin))
    if remote.get('evaluations'):
        path=out/'evaluations.jsonl';tmp=path.with_suffix('.dispatch.tmp')
        tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in remote['evaluations']));tmp.replace(path)
        evaluations=out/'evaluations';evaluations.mkdir(exist_ok=True)
        for r in remote['evaluations'][-7:]:atomic(evaluations/f"{r['steps']:09d}.json",r)
    if remote.get('verified'):
        r=remote['verified']
        assert r['completed'] and r['steps']==5000192 and r['updates']==623424 and r['backend']=='gpu'
        assert r['entropy_search']['value']==j['value'] and r['seed']==j['seed']
        atomic(out/'verified-completion.json',dict(r,remote_origin=origin));j['result']=r
        j['status']='completed';j['finished_unix']=r['finished_unix']
    if remote.get('failure'):
        atomic(out/'failure.json',dict(remote['failure'],remote_origin=origin));j['error']=remote['failure'].get('error');j['status']='failed'


def tick(plan,promote,job_factory):
    now=time.time()
    if plan['dispatch_role']=='master':
        if now-plan.get('last_dispatch_poll',0)>=20:
            errors={}
            for peer in plan['dispatch_peers']:
                try:
                    data=read_peer(plan,peer);lookup={j['id']:j for j in data['jobs']}
                    for j in plan['jobs']:
                        if j.get('delegation',{}).get('host')==peer['id'] and j['id'] in lookup:
                            mirror(plan,j,lookup[j['id']],peer['id'])
                except Exception as error:errors[str(peer['id'])]=repr(error)
            plan.update(dispatch_errors=errors,last_dispatch_poll=now)
        promote(plan)
        for j in plan['jobs']:
            if j['phase']=='final' and not j.get('delegation'):mark_delegated(plan,j)
    else:
        if now-plan.get('last_dispatch_poll',0)<20:return
        try:
            data=read_peer(plan,plan['dispatch_peers'][0]);selections=data['selections']
            plan.update(master_selections=selections,last_dispatch_poll=now,dispatch_errors={})
            existing={j['id'] for j in plan['jobs']}
            for task in ('slide','reach'):
                selected=selections.get(task+'/meow',{})
                if selected.get('status') not in ('selected','selected_unvalidated_all_zero'):continue
                value=selected['best']['value']
                for seed in plan['dispatch_final_seeds']:
                    j=job_factory(plan,'meow',task,value,seed,'final')
                    if j['id'] not in existing:
                        plan['jobs'].append(j);existing.add(j['id'])
        except Exception as error:
            plan.update(last_dispatch_poll=now,dispatch_errors={'master':repr(error)})


def finish_ready(plan):
    if plan['dispatch_role']=='master':return True
    for task in ('slide','reach'):
        selected=plan.get('master_selections',{}).get(task+'/meow',{})
        if selected.get('status')=='no_valid_pair':continue
        if selected.get('status') not in ('selected','selected_unvalidated_all_zero'):return False
        finals=[j for j in plan['jobs'] if j['phase']=='final' and j['task']==task]
        if sorted(j['seed'] for j in finals)!=sorted(plan['dispatch_final_seeds']):return False
    return True


def build_controller(original):
    source=original.replace('import time\n','import time\nfrom fetch_meow_dispatch import tick,finish_ready\n',1)
    needle="        for j in plan['preflights']+plan['jobs']:\n            if j['status'] not in ('initializing','running'):continue"
    assert source.count(needle)==1
    source=source.replace(needle,"        for j in plan['preflights']+plan['jobs']:\n            if j.get('delegation'):continue\n            if j['status'] not in ('initializing','running'):continue")
    assert source.count('update();promote(plan)')==1
    source=source.replace('update();promote(plan)','update();tick(plan,promote,job)')
    needle="active=[j for j in plan['jobs']+plan['preflights'] if j['status'] in ('running','initializing')]"
    assert source.count(needle)==1
    source=source.replace(needle,"active=[j for j in plan['jobs']+plan['preflights'] if not j.get('delegation') and j['status'] in ('running','initializing')]")
    needle="                    if j['status']!='queued':continue"
    assert source.count(needle)==1
    source=source.replace(needle,"                    if j.get('delegation'):continue\n"+needle)
    needle="            ready=all(j['status']=='completed' for j in plan['preflights'])"
    assert source.count(needle)==1
    source=source.replace(needle,needle+"\n            if plan.get('runtime_ready_file') and not Path(plan['runtime_ready_file']).exists():\n                plan.update(status='waiting_for_cuda_runtime',checked_unix=time.time());atomic(path,plan);time.sleep(10);continue")
    needle="if ready and all(j['status'] in ('completed','failed') for j in plan['jobs']):"
    assert source.count(needle)==1
    source=source.replace(needle,"if ready and finish_ready(plan) and all(j['status'] in ('completed','failed') for j in plan['jobs']):")
    ast.parse(source)
    return source


def reader_main(config):
    cfg=json.loads(Path(config).read_text());plan=json.loads(Path(cfg['plan']).read_text())
    result=dict(campaign=plan['campaign'],host=plan['host'])
    if cfg['mode']=='selection':
        result['selections']={k:v for k,v in plan['selections'].items() if k.endswith('/meow')}
    else:
        rows=[]
        for j in plan['jobs']:
            assert j['method']=='meow' and j['task'] in ('slide','reach') and j['phase'] in ('pilot','final')
            out=Path(j['output']);r={k:j[k] for k in ('id','method','task','seed','phase','value','status','output')}
            for key,name in [('progress','progress.json'),('verified','verified-completion.json'),('failure','failure.json'),('config','config.json'),('entropy_audit','entropy-parameter-audit.json')]:
                try:r[key]=json.loads((out/name).read_text())
                except (OSError,ValueError):pass
            if r.get('verified'):
                c=json.loads((out/'config.json').read_text())
                assert c['num_envs']==256 and c['batch_size']==256 and c['utd']==.125 and c['HER'] and c['normalize_observations']
            values=[];f=out/'evaluations.jsonl'
            if f.exists():
                for line in f.read_text().splitlines():
                    try:
                        e=json.loads(line);primary=e['primary_mode'];a=e[primary]
                        values.append(dict(steps=e['steps'],updates=e['updates'],primary_mode=primary,
                            **{primary:{k:a[k] for k in ('episodes','success_rate','any_success_rate','mean_return')}}))
                    except (ValueError,KeyError):pass
            r['evaluations']=values;rows.append(r)
        result['jobs']=rows
    print(json.dumps(result,separators=(',',':'),allow_nan=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reader-config',required=True)
    args=parser.parse_args();reader_main(args.reader_config)
