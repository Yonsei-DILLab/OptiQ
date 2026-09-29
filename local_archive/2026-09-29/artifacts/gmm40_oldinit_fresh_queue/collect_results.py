"""Read-only monitor; collect and verify the approved campaign's final archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

HERE=Path(__file__).resolve().parent
HOST='vast-heechan-180'
REMOTE_ROOT='/home/heechan/optiq-experiments/gmm40-trg-oldinit-fresh-100k-4seed-20260921'
SSH=['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',HOST]
SCRIPT=r'''
import json,time
from pathlib import Path
r=Path('/home/heechan/optiq-experiments/gmm40-trg-oldinit-fresh-100k-4seed-20260921')
read=lambda p:json.loads(p.read_text()) if p.exists() else None
out={k:read(r/(k+'.json')) for k in ['manifest','status','failure','preflight','result']}
out['jobs']=[]
for j in out['manifest']['jobs']:
    status=read(r/'jobs'/(j['name']+'.json'))
    progress=read(r/'results'/j['name']/'status.json')
    item={'name':j['name'],'method':j['method'],'seed':j['seed'],'queue':status,'progress':progress}
    if status['status']=='failed':
        log=r/'logs'/(j['name']+'.log')
        if log.exists():item['log_tail']=log.read_text(errors='replace')[-10000:]
    out['jobs'].append(item)
out['wandb_repair_status']=read(r/'wandb-repair-status.json')
out['wandb_repair_failure']=read(r/'wandb-repair-failure.json')
out['wandb_repairs']={j['name']:read(r/'results'/j['name']/'wandb_repair.json') for j in out['manifest']['jobs']}
out['checked_at']=time.time()
print(json.dumps(out))
'''


def main():
    response=subprocess.run(SSH+['python3 -'],input=SCRIPT,text=True,capture_output=True,timeout=60,check=True)
    data=json.loads(response.stdout)
    plan=data['manifest']['plan']
    assert plan['steps']==100000 and plan['seeds']==[0,1,2,3] and len(data['manifest']['jobs'])==4
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    for key in ('manifest','status','failure','preflight','result'):
        if data[key] is not None:(dest/f'{key}.json').write_text(json.dumps(data[key],indent=2)+'\n')
    (HERE/'monitor-latest.json').write_text(json.dumps(data,indent=2)+'\n')
    (dest/'wandb-repair.json').write_text(json.dumps({k:data[k] for k in
        ['wandb_repair_status','wandb_repair_failure','wandb_repairs']},indent=2)+'\n')
    result=data.get('result')
    if result and result['status']=='completed':
        assert result['runs']==4 and all(j['queue']['status']=='completed' for j in data['jobs'])
        archive=dest/'final-results.tar.gz'
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            return h.hexdigest()
        if not archive.exists() or digest(archive)!=result['archive_sha256']:
            temp=dest/'final-results.tar.gz.part'
            subprocess.run(['scp','-o','ConnectTimeout=10',HOST+':'+result['archive'],str(temp)],check=True,timeout=900)
            assert digest(temp)==result['archive_sha256']
            temp.replace(archive)
        with tarfile.open(archive) as tar:
            tar.extractall(dest,filter='data')
        (dest/'LOCAL_COPY_VERIFIED.json').write_text(json.dumps(dict(sha256=result['archive_sha256'],verified_at=time.time()),indent=2)+'\n')
    print(json.dumps(dict(phase=(data['status'] or {}).get('phase'),
        jobs=[dict(name=j['name'],status=j['queue']['status'],step=(j['progress'] or {}).get('step',0),gpu=j['queue'].get('gpu')) for j in data['jobs']],
        failure=data['failure'],wandb_repair_status=data['wandb_repair_status'],
        wandb_repair_failure=data['wandb_repair_failure'],
        local_final_archive_verified=(dest/'LOCAL_COPY_VERIFIED.json').exists()),indent=2))


if __name__=='__main__':main()
