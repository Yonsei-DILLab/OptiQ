"""Apply the committed Q screen only after three verified post500k policy evaluations."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess

ROOT=Path(__file__).resolve().parent
SOURCE='a5de32fe885b79700ab4566bf36fe91de2ee3860'
REMOTE='/home/heechan/optiq-experiments/antmaze-optiq-v3-teacherfloor1-retention-1m-s0-20260925'
JOB='v3-optiq-startnorm-geodesic-T3-teacherfloor1-1m-s0'

REMOTE_CODE=r'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path(REQ['root']);job=REQ['job'];source=REQ['source']
manifest=json.loads((root/'manifest.json').read_text())
assert manifest['source_commit']==source and len(manifest['jobs'])==1
assert manifest['jobs'][0]['id']==job
assert not (root/'screen-stop.json').exists()
state=json.loads((root/'status.json').read_text())
assert not state['pending'] and len(state['running'])==1 and not state['completed']
for row in REQ['evaluations']:
 p=root/'runs'/job/'evaluations'/f"{row['step']:010d}"/'policy-fixed/rollouts.npz'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
pid=int(subprocess.check_output(ctl+['pid',root.name],text=True).strip())
assert pid==560894 and os.getpgid(pid)==pid
actual=[]
for identifier in (560894,560959,561664):
 p=Path('/proc')/str(identifier)
 assert p.exists() and os.getpgid(identifier)==pid
 command=(p/'cmdline').read_bytes().decode().replace('\0',' ')
 assert str(root) in command and str((p/'cwd').resolve()).endswith(source)
 actual.append(dict(pid=identifier,pgid=pid,cmdline=command,cwd=str((p/'cwd').resolve())))
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
backup=root/('screen-stop-backup-'+stamp);backup.mkdir()
for p in (root/'manifest.json',root/'status.json',root/'jobs'/(job+'.json'),root/'runs'/job/'progress.json'):
 shutil.copy2(p,backup/(('job-' if p.parent.name=='jobs' else '')+p.name))
subprocess.run(ctl+['stop',root.name],check=True,capture_output=True)
subprocess.run(ctl+['stop',root.name+'-wandb-sync'],check=True,capture_output=True)
members=[]
for line in subprocess.check_output(['ps','-eo','pid=,pgid=,args='],text=True).splitlines():
 fields=line.strip().split(None,2)
 if len(fields)>=2 and int(fields[1])==pid:members.append(line)
assert not members,members
progress=json.loads((root/'runs'/job/'progress.json').read_text())
evidence=dict(id=job,last_logged_step=progress['step'],updates=progress['updates'],evaluations=REQ['evaluations'])
record=dict(time=time.time(),source_commit=source,reason=REQ['reason'],protocol=REQ['protocol'],
 completed=False,stopped_early=True,planned_budget=manifest['jobs'][0]['steps'],
 evidence=[evidence],actual_processes_before=actual,process_group_empty=True,
 backup=str(backup),preserved='All frozen source, logs, raw evaluations and intermediate policies; no final1M claim')
(root/'screen-stop.json').write_text(json.dumps(record,indent=2)+'\n')
state.update(time=time.time(),running=[],pending=[],stopped_early=[job],screen_stop=record)
(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
jp=root/'jobs'/(job+'.json');jd=json.loads(jp.read_text());jd.update(status='stopped_early',screen_stop=record)
jp.write_text(json.dumps(jd,indent=2)+'\n')
print(json.dumps(record))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    report=json.loads((ROOT/'report/results.json').read_text())
    assert report['source_commit']==SOURCE and not report['planned_long_budget_completed']
    rows=sorted([r for r in report['history'] if r['condition']=='floor1-long' and r['mode']=='policy' and r['step']>=500000],key=lambda r:r['step'])[-3:]
    assert len(rows)==3,'Three post500k direct evaluations required'
    assert all(49000<=b['step']-a['step']<=51000 for a,b in zip(rows,rows[1:]))
    for row in rows:
        assert row['source_commit']==SOURCE and row['episodes']==40 and row['id']==JOB
        assert row['route_counts'].get('left',0)<=1 and row['successful_route_counts'].get('left',0)==0
        assert row['fixed_original_start'] and row['reward_telescope_verified']
        assert hashlib.sha256(Path(row['raw_path']).read_bytes()).hexdigest()==row['raw_sha256']
    req=dict(root=REMOTE,job=JOB,source=SOURCE,protocol='antmaze_experiments/TEACHER_FLOOR_RETENTION_PROTOCOL.md',
             reason='Three consecutive direct-policy evaluations from500k have at most1/40 left entries and zero left successes',
             evaluations=[{k:r[k] for k in ('step','episodes','route_counts','successful_route_counts','raw_sha256')} for r in rows])
    (ROOT/'screen-request.json').write_text(json.dumps(req,indent=2)+'\n')
    print(json.dumps(req),flush=True)
    if args.execute:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','vast-heechan-180','python3','-'],
                              input=('REQ='+repr(req)+'\n'+REMOTE_CODE).encode(),capture_output=True,check=True,timeout=90)
        record=json.loads(result.stdout)
        (ROOT/'screen-stop-execution.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
