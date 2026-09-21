"""Audited, checkpoint-preserving drain and idempotent launch of two lanes."""
from pathlib import Path
import argparse, hashlib, json, subprocess, time
import support as rt
from worker import REVISIONS

OLD_NAMES={'nsq-resume-main','nsq-resume-legacy'}
PARTITIONS='big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000'

def jobs():
    for attempt in range(10):
        p=subprocess.run(['squeue','-h','-r','-u','hobbit9882','-o','%i|%j|%T'],capture_output=True,text=True,timeout=30)
        if p.returncode==0 and p.stdout.strip():
            return [dict(zip(('id','name','state'),line.split('|'))) for line in p.stdout.splitlines()]
        time.sleep(min(20,2+attempt*2))
    raise RuntimeError('No reliable scheduler snapshot; no mutations made from an empty response')

def capture(base,ids):
    rows=[]
    for revision in REVISIONS:
        root=base/revision
        for p in (root/'queue').glob('*.json'):
            lease=json.loads(p.read_text())
            if lease.get('job_id') not in ids:continue
            out=root/'runs'/p.stem
            progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
            rows.append(dict(root=str(root),name=p.stem,job=lease['job_id'],pre_step=progress.get('step',0)))
    return rows

def prepare(base,ops):
    record=ops/'DRAIN.json'
    if record.exists():raise RuntimeError('Drain already recorded; inspect it, do not signal twice blindly')
    old=[j for j in jobs() if j['name'] in OLD_NAMES]
    before=capture(base,{j['id'] for j in old})
    rt.write(record,dict(time=time.time(),old_jobs=old,checkpoints_before=before,phase='preparing'))
    pending=[j['id'] for j in old if j['state']=='PENDING']
    if pending:subprocess.run(['scancel','--state=PENDING',*pending],check=True)
    current=[j for j in jobs() if j['name'] in OLD_NAMES]
    pending=[j for j in current if j['state']=='PENDING']
    if pending:raise RuntimeError('Pending replacements remain; inspect before continuing')
    live=[j['id'] for j in current if j['state']=='RUNNING']
    more=capture(base,set(live))
    known={(r['root'],r['name']) for r in before}
    before.extend(r for r in more if (r['root'],r['name']) not in known)
    rt.write(record,dict(time=time.time(),old_jobs=old,checkpoints_before=before,signalled=live,phase='signalling'))
    if live:subprocess.run(['scancel','--signal=TERM','--batch',*live],check=True)
    rt.write(record,dict(time=time.time(),old_jobs=old,checkpoints_before=before,signalled=live,phase='draining'))
    print(json.dumps(dict(signalled=live,cancelled_pending=len([j for j in old if j['state']=='PENDING']),checkpoint_runs=len(before))),flush=True)

def verify(base,ops):
    old=[j for j in jobs() if j['name'] in OLD_NAMES]
    if old:raise RuntimeError('Old workers not yet drained: '+repr(old))
    record=json.loads((ops/'DRAIN.json').read_text());verified=[]
    import msgpack
    for r in record['checkpoints_before']:
        root=Path(r['root']);out=root/'runs'/r['name'];cp=out/'checkpoint.msgpack'
        assert cp.exists(),str(cp)
        blob=cp.read_bytes();ck=msgpack.unpackb(blob,raw=False,strict_map_key=False)
        expected=json.loads((root/'SOURCE_MANIFEST.json').read_text())['code_id']
        assert ck['source_code_id']==expected and int(ck['step'])>=r['pre_step'],r
        assert all(k in ck for k in ('actor','key','collectkey','train_total','elapsed')),r
        if '_closed_' in r['name'] or '_source_' in r['name']:
            assert all(k in ck for k in ('critic','replay','env_state','episode_step')),r
        verified.append(dict(**r,saved_step=int(ck['step']),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),complete=(out/'COMPLETE.json').exists()))
    rt.write(ops/'DRAIN_VERIFIED.json',dict(time=time.time(),checkpoints=verified,old_workers_remaining=0))
    print(json.dumps(dict(verified=len(verified),steps=[r['saved_step'] for r in verified])),flush=True)

def launch(base,ops,event):
    assert (ops/'DRAIN_VERIFIED.json').exists(),'Verify checkpoint drain before launching'
    assert not [j for j in jobs() if j['name'] in OLD_NAMES],'Old worker overlap'
    record=ops/'SUBMISSIONS.json';rows=json.loads(record.read_text()) if record.exists() else []
    (ops/'logs').mkdir(exist_ok=True)
    for lane,count in [('fast',18),('exact',2)]:
        if any(r['lane']==lane for r in rows):continue
        cmd=['sbatch','--parsable','--partition='+PARTITIONS,'--qos=big_qos',f'--array=0-{count-1}%{count}',
             '--job-name=nsq-lane-'+lane,'--output='+str(ops/'logs'/(lane+'-%A_%a.out')),
             str(ops/'job.sbatch'),str(base),str(ops),lane,event]
        p=subprocess.run(cmd,capture_output=True,text=True,check=True);job=p.stdout.strip().split(';')[0];assert job.isdigit()
        rows.append(dict(lane=lane,workers=count,job_id=job,time=time.time(),command=cmd,operations_commit=json.loads((ops/'OPS_DEPLOYMENT.json').read_text())['commit'],numerical_commits=REVISIONS))
        rt.write(record,rows);print(json.dumps(rows[-1]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','verify','launch']);p.add_argument('--base',required=True);p.add_argument('--ops',required=True);p.add_argument('--event',default='lanes_20260919')
    a=p.parse_args();base=Path(a.base).resolve();ops=Path(a.ops).resolve()
    if a.phase=='prepare':prepare(base,ops)
    elif a.phase=='verify':verify(base,ops)
    else:launch(base,ops,a.event)
