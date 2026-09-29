"""Read-only archive of the eight completed DACER runs and their evaluations."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent / 'results'
REMOTE = r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
task=sys.argv[1]
root=Path('/home/heechan/optiq-experiments/trg-dacer-priority-20260921')
result=json.loads((root/f'result-{task}.json').read_text())
files=[]
for key in ('status','failure','selection','result','manifest','beta-cancellation'):
    p=root/f'{key}-{task}.json'
    if p.exists(): files.append((p,Path(p.name)))
jobs=[j for j in result['jobs'] if j['stage']=='dacer']
assert len(jobs)==4 and {j['seed'] for j in jobs}==set(range(4))
for job in jobs:
    assert job['status']=='completed' and job['commit']=='efe67fde083513b89f57a34a54743cb1e6d4b49d'
    d=Path(job['run_dir']);dest=Path('runs')/job['name']
    files.append((root/'jobs'/(job['name']+'.json'),Path('jobs')/(job['name']+'.json')))
    files.append((d/'config.json',dest/'config.json'))
    for mode in ('zero_z','stochastic_z'):
        p=list(d.rglob('evaluations_'+mode+'.npz'));assert len(p)==1
        files.append((p[0],dest/p[0].name))
    for p in d.glob('*.json'):
        if p.name!='config.json': files.append((p,dest/p.name))
audit={str(name):hashlib.sha256(p.read_bytes()).hexdigest() for p,name in files}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for p,name in files: tar.add(p,arcname=str(name))
    data=json.dumps(audit,indent=2).encode();info=tarfile.TarInfo('SHA256.json');info.size=len(data)
    tar.addfile(info,io.BytesIO(data))
'''

def collect(pair):
    host,task=pair
    dest=ROOT/task;dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'evaluations-and-config.tar.gz'
    with archive.open('wb') as out:
        p=subprocess.run(['ssh','-o','ConnectTimeout=15',host,'python3 - '+task],input=REMOTE.encode(),stdout=out,stderr=subprocess.PIPE,timeout=120)
    if p.returncode: raise RuntimeError(p.stderr.decode())
    with tarfile.open(archive) as tar: tar.extractall(dest,filter='data')
    audit=json.loads((dest/'SHA256.json').read_text())
    for name,digest in audit.items(): assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,name
    (dest/'LOCAL_COPY_VERIFIED.json').write_text(json.dumps(dict(host=host,files=len(audit),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2)+'\n')
    return dict(task=task,verified_files=len(audit),destination=str(dest))

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for result in pool.map(collect,[('vast-heechan-180','humanoid'),('vast-heechan-199','ant')]): print(json.dumps(result))
