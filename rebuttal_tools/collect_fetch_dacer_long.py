"""Collect the four-host study and produce matched-budget five-seed reports."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import tempfile
import time
import zlib


QUERY=r'''
import pathlib,json,time,base64,zlib
base=pathlib.Path(BASE);plan=json.loads((base/'plan.json').read_text())
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return None
def alive(pid,ticks):
 try:
  s=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and int(s[19])==ticks
 except OSError:return False
rows=[]
for j in plan['jobs']:
 out=pathlib.Path(j['output']);progress=read(out/'progress.json');endpoints={};exports={}
 for step in [5000192,10000128]:
  f=out/'evaluations'/f'{step:09d}.json'
  record=read(f)
  if record and record[record['primary_mode']]['episodes']==500:
   v=record[record['primary_mode']]
   endpoints[str(step)]=dict(success=v['success_rate'],episodes=v['episodes'],mean_return=v['mean_return'],stages=v['stages'],updates=record['updates'])
   if j['id']+'-'+str(step) not in KNOWN_ENDPOINTS:exports[str(step)]=record
 recent=[]
 for f in sorted((out/'evaluations').glob('*.json'))[-8:]:
  r=read(f)
  if not r:continue
  v=r[r['primary_mode']]
  if v['episodes']==10:recent.append(dict(step=r['steps'],success=v['success_rate'],air=v['stages']['air_over5cm']))
 late=recent[-5:]
 row={k:j.get(k) for k in ['id','method','profile','seed','phase','status','pid','start_ticks','gpu','learner_cpus','worker_cpus','output','source','source_manifest','target_entropy_per_dim','noise_scale']}
 row.update(live=alive(j.get('pid'),j.get('start_ticks')),progress=progress,endpoints=endpoints,
  last5=late,failure=read(out/'failure.json'),verified=read(out/'verified-completion.json'))
 if exports:row['endpoint_export']=exports
 if row['verified'] and j['id'] not in KNOWN_CURVES:
  records={r['steps']:r for r in map(json.loads,(out/'evaluations.jsonl').read_text().splitlines())}
  curve=[]
  for step,r in sorted(records.items()):
   v=r[r['primary_mode']];air=v['stages']['air_over5cm']
   curve.append([step,r['updates'],v['episodes'],v['success_rate'],v['any_success_rate'],air['final_successes'],air['episodes']])
  row['curve_export']=dict(columns=['steps','updates','episodes','success','any_success','air_successes','air_episodes'],rows=curve)
 rows.append(row)
result=dict(host=plan['host'],status=plan['status'],error=plan.get('error'),controller_pid=plan.get('controller_pid'),
 source=plan['source'],source_commit=plan['source_commit'],native_files_verified=plan['native_files_verified'],
 cpu_checks_passed=plan.get('cpu_checks_passed'),gpu_checks_passed=plan.get('gpu_checks_passed'),
 checked_unix=time.time(),selection=plan.get('selection'),admission=plan.get('admission'),
 preflights=[dict(profile=j['profile'],status=j['status'],error=j.get('error'),result=j.get('result')) for j in plan['preflights']],jobs=rows)
print(base64.b64encode(zlib.compress(json.dumps(result,separators=(',',':')).encode())).decode())
'''


def stats(values):
    return dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values) if len(values)>1 else None,values=values)


def collect(root,socket_dir):
    protocol=json.loads((root/'protocol.json').read_text());hosts={}
    for name in ('curves','endpoint-evaluations'):(root/name).mkdir(exist_ok=True)
    def query(i):
        meta=root/f'host{i}-deployment.json'
        if not meta.exists():return i,dict(host=i,connection_error='Deployment not yet recorded')
        deployment=json.loads(meta.read_text());base=str(Path(deployment['plan']).parent)
        known_curves=[f.stem.removeprefix(f'host{i}-') for f in (root/'curves').glob(f'host{i}-*.json')]
        known_endpoints=[f.stem.removeprefix(f'host{i}-') for f in (root/'endpoint-evaluations').glob(f'host{i}-*.json')]
        code='BASE='+repr(base)+'\nKNOWN_CURVES='+repr(known_curves)+'\nKNOWN_ENDPOINTS='+repr(known_endpoints)+'\n'+QUERY
        args=['ssh','-T','-n','-o','IPQoS=none','-o','BatchMode=yes','-o','ConnectTimeout=8',
            '-o','ControlMaster=auto','-o','ControlPersist=300','-o','ControlPath='+str(socket_dir/f'host{i}'),
            'ibolt-server-'+str(i),'/usr/bin/python3 -I -S -c '+shlex.quote(code)]
        try:
            r=subprocess.run(args,capture_output=True,text=True,timeout=40)
            if r.returncode:return i,dict(host=i,connection_error=r.stderr[-700:])
            d=json.loads(zlib.decompress(base64.b64decode(r.stdout.strip())))
            for j in d['jobs']:
                if 'curve_export' in j:
                    f=root/'curves'/f'host{i}-{j["id"]}.json';f.write_text(json.dumps(j.pop('curve_export'))+'\n')
                for step,record in j.pop('endpoint_export',{}).items():
                    f=root/'endpoint-evaluations'/f'host{i}-{j["id"]}-{step}.json';f.write_text(json.dumps(record)+'\n')
            (root/f'host{i}-latest.json').write_text(json.dumps(d,indent=2)+'\n');return i,d
        except (subprocess.TimeoutExpired,ValueError) as error:return i,dict(host=i,connection_error=repr(error))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,d in pool.map(query,[1,2,3,4]):hosts[str(i)]=d
    jobs=[dict(j,host=int(i)) for i,d in hosts.items() for j in d.get('jobs',[])]
    comparisons=[];controls=sorted([j for j in jobs if j['method']=='sac'],key=lambda j:j['seed'])
    control_ready=len(controls)==5 and [j['seed'] for j in controls]==[0,1,2,3,4] and all(j['verified'] for j in controls)
    winners=[]
    for i,d in hosts.items():
        selection=d.get('selection')
        if selection:
            winner=selection['ranking'][0];winners.append(dict(winner,host=int(i)))
    for winner in winners:
        profile=winner['profile'];group=sorted([j for j in jobs if j['profile']==profile and j['method']=='ibolt'],key=lambda j:j['seed'])
        complete=len(group)==5 and [j['seed'] for j in group]==[0,1,2,3,4] and all(j['verified'] for j in group)
        report=dict(profile=profile,host=winner['host'],pilot_ranking=winner,complete=bool(complete and control_ready))
        if report['complete']:
            for step in ['5000192','10000128']:
                iv=[j['endpoints'][step]['success'] for j in group];sv=[j['endpoints'][step]['success'] for j in controls]
                gap=[x-y for x,y in zip(iv,sv)]
                air={}
                for label,rows in [('ibolt',group),('sac',controls)]:
                    values=[j['endpoints'][step]['stages']['air_over5cm'] for j in rows]
                    air[label]=dict(successes=sum(v['final_successes'] for v in values),episodes=sum(v['episodes'] for v in values))
                report[step]=dict(ibolt=stats(iv),sac=stats(sv),paired_gap=stats(gap),additional_seeds_gap=stats(gap[2:]),air=air,
                    observed_large_win=statistics.mean(gap)>=.1-1e-12 and statistics.mean(gap[2:])>1e-12)
        comparisons.append(report)
    global_selected=max(winners,key=lambda w:tuple(w['rank']))['profile'] if len(winners)==4 else None
    terminal={'completed','completed_with_failures','controller_error','stopped'}
    finished=len(hosts)==4 and all(d.get('status') in terminal and not d.get('connection_error') for d in hosts.values())
    result=dict(checked_unix=time.time(),campaign=protocol['campaign'],protocol=protocol,
        counts=dict(Counter(j['status'] for j in jobs)),active=sum(j['live'] for j in jobs),
        connection_errors={i:d['connection_error'] for i,d in hosts.items() if d.get('connection_error')},
        global_pilot_selected_profile=global_selected,comparisons=comparisons,finished=finished,hosts=hosts,
        interpretation='Prespecified pilot selection and all local confirmations are reported. Observed margins are not statistical significance tests. Search uses more iBOLT-specific tuning than the unchanged SAC control.')
    tmp=root/'study-status.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'study-status.json')
    if finished:(root/'final-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checked_unix=result['checked_unix'],counts=result['counts'],active=result['active'],
        connection_errors=result['connection_errors'],global_pilot_selected_profile=global_selected,
        comparisons=[dict(profile=r['profile'],complete=r['complete']) for r in comparisons],finished=finished)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    root=a.directory or Path(Path('/tmp/ibolt-fetch-dacer-long-path').read_text().strip())
    sockets=Path(tempfile.mkdtemp(prefix='fetch-long-ssh-'));os.chmod(sockets,0o700)
    if a.watch:
        (root/'monitor-pid.json').write_text(json.dumps(dict(pid=os.getpid(),watch=True,socket_dir=str(sockets),started_unix=time.time()))+'\n')
    while True:
        result=collect(root,sockets)
        if not a.watch or result['finished']:
            if a.watch and (root/'monitor-service.json').exists():
                service=json.loads((root/'monitor-service.json').read_text());plist=Path(service['plist'])
                if plist.parent==Path.home()/'Library/LaunchAgents' and plist.name==service['label']+'.plist':plist.unlink(missing_ok=True)
            return
        time.sleep(60)


if __name__=='__main__':main()
