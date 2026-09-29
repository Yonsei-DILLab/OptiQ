"""Download existing lightweight evaluation results; never run training/evaluation."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

OUT = Path(__file__).resolve().parent
HOSTS = ('vast-heechan-180',)
ROOT = '/home/heechan/optiq-experiments/antmaze-optiq-v3-gamma999-T3-retention-1m-s0-20260925'
CODE = r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile,time,zipfile
root=Path(__ROOT__)
metadata={'root':str(root),'time':time.time(),'read_only':True,'files':{},'skipped':{},'unchanged':{}}
paths=set()
for name in ('manifest','registration','status','failure','result','wandb-sync-status',
             'controller-provenance','priority-dispatch-audit','source-sharing','screen-stop','wandb-screen-annotation'):
    p=root/(name+'.json')
    if p.exists():paths.add(p)
for sub in ('jobs','proofs','stages'):
    paths.update((root/sub).glob('*.json'))
names=('config','result','progress','parameter-audit','dynamics-verification',
       'checkpoint-verification','optiq-profile-verification','dacer-target-verification','dacer_regulator',
       'failure','wandb')
for phase in ('preflight','runs'):
    for run in (root/phase).glob('*'):
        if not run.is_dir():continue
        for name in names:
            p=run/(name+'.json')
            if p.exists():paths.add(p)
        history=run/'dacer_regulator_history.jsonl'
        if history.exists():paths.add(history)
        paths.update(run.glob('*proof*.json'))
        paths.update(run.glob('*verification*.json'))
        paths.update(run.glob('policy-checkpoints/*/verification.json'))
        if phase=='runs':
            paths.update(run.glob('replay-diagnostics/*/summary.json'))
            for summary in run.glob('evaluations/*/*/summary.json'):
                try:
                    json.loads(summary.read_bytes())
                    rollouts=summary.parent/'rollouts.npz'
                    if not rollouts.is_file() or not zipfile.is_zipfile(rollouts):
                        metadata['skipped'][str(summary.parent.relative_to(root))]='rollout incomplete/missing'
                        continue
                except (OSError,ValueError):
                    metadata['skipped'][str(summary.parent.relative_to(root))]='summary incomplete'
                    continue
                paths.update((summary,rollouts))
                proof=summary.parent/'critic-diagnostics.json'
                if proof.is_file():paths.add(proof)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
    for p in sorted(paths):
        key=str(p.relative_to(root))
        try:
            before=p.stat()
            limit=50*1024*1024 if p.suffix=='.npz' else 2*1024*1024
            if before.st_size>limit:
                metadata['skipped'][key]='size limit'
                continue
            content=p.read_bytes()
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                metadata['skipped'][key]='changed during read'
                continue
            if p.suffix=='.json':json.loads(content)
            if p.suffix=='.npz':
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    if z.testzip() is not None:raise ValueError('ZIP CRC mismatch')
            digest=hashlib.sha256(content).hexdigest()
            if KNOWN.get(key)==digest:
                metadata['unchanged'][key]=digest
                continue
            entry=tarfile.TarInfo(key);entry.size=len(content);entry.mtime=before.st_mtime
            tar.addfile(entry,io.BytesIO(content))
            metadata['files'][key]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'mtime':before.st_mtime}
        except (OSError,ValueError,zipfile.BadZipFile) as exc:
            metadata['skipped'][key]=type(exc).__name__
    content=json.dumps(metadata,indent=2).encode()
    entry=tarfile.TarInfo('collection-metadata.json');entry.size=len(content)
    tar.addfile(entry,io.BytesIO(content))
'''.replace('__ROOT__', repr(ROOT))


def collect(host, stamp, job=None):
    if host not in HOSTS:
        raise ValueError('Unapproved host')
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
             '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',
             '-o','ControlMaster=no','-o','ControlPath=none',host,'python3','-']
    previous=OUT/'results'/host
    known={str(p.relative_to(previous)):hashlib.sha256(p.read_bytes()).hexdigest() for p in previous.rglob('*') if p.is_file() and p.name!='collection-metadata.json'} if previous.exists() else {}
    code='KNOWN='+repr(known)+'\n'+CODE
    if job is not None:
        if '..' in job or not job.replace('-','').replace('_','').replace('.','').isalnum():
            raise ValueError('Unsafe job identifier')
        code=code.replace("with tarfile.open(fileobj=sys.stdout.buffer", 
            "paths={p for p in paths if p.parent==root or "+repr(job)+" in p.relative_to(root).parts or p.stem=="+repr(job)+"}\nwith tarfile.open(fileobj=sys.stdout.buffer")
    p=subprocess.run(command,input=code.encode(),capture_output=True,timeout=90,check=True)
    destination=OUT/'results'/host
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
        files={}
        for member in tar.getmembers():
            relative=PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive member')
            files[member.name]=tar.extractfile(member).read()
    metadata=json.loads(files['collection-metadata.json'])
    for relative,record in metadata['files'].items():
        content=files[relative]
        assert hashlib.sha256(content).hexdigest()==record['sha256']
        assert len(content)==record['bytes']
    archives=OUT/'result-collections'/stamp
    archives.mkdir(parents=True,exist_ok=True)
    for relative,content in files.items():
        if job is not None and relative=='collection-metadata.json':
            relative='collection-metadata-'+job+'.json'
        path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.download.',delete=False) as handle:
            handle.write(content)
            temporary=Path(handle.name)
        temporary.replace(path)
    label=host+('-'+job if job is not None else '')
    (archives/(label+'-metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    result={'host':host,'destination':str(destination),'files':len(metadata['files']),
            'rollout_files':sum(n.endswith('/rollouts.npz') for n in metadata['files']),
            'critic_diagnostics':sum(n.endswith('/critic-diagnostics.json') for n in metadata['files']),
            'archive_bytes':len(p.stdout),'archive_sha256':hashlib.sha256(p.stdout).hexdigest(),
            'skipped':metadata['skipped']}
    result['job']=job
    (archives/(label+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(collect,host,stamp):host for host in HOSTS}
        results=[]
        for future in concurrent.futures.as_completed(futures):
            host=futures[future]
            try:result=future.result()
            except (ValueError,OSError,subprocess.SubprocessError) as exc:
                result={'host':host,'error_type':type(exc).__name__,'returncode':getattr(exc,'returncode',None)}
            results.append(result)
            print(json.dumps(result),flush=True)
    (OUT/'latest-result-collection.json').write_text(json.dumps({'time_utc':stamp,'hosts':results},indent=2)+'\n')
    return int(any('error_type' in x for x in results))


if __name__=='__main__':
    raise SystemExit(main())
