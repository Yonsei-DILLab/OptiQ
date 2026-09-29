#!/usr/bin/env python3
"""Read-only, bounded JSON snapshots of the two critic-dynamics campaign shards.

Run only after the campaign has been registered. No remote files, jobs, services,
or model state are changed. Checkpoints, replay buffers, NPZs and logs are not
downloaded. Missing/partially-written JSON is explicitly recorded, not treated
as a successful or zero-valued result.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


CAMPAIGN = 'antmaze-optiq-critic-dynamics-500k-s0-20260925'
ROOT = '/home/heechan/optiq-experiments/' + CAMPAIGN
HOSTS = ('vast-heechan-180', 'vast-heechan-199')
OUT = Path(__file__).resolve().parent

# Use only the system Python standard library on the remote host. Deliberately
# do not import the training package, initialize a GPU, or inspect credentials.
REMOTE_CODE = r'''
import hashlib,json,re,time
from pathlib import Path
root=Path(__ROOT__)
limit=2*1024*1024
metadata={}
def reject_nonfinite(value):
    raise ValueError('Non-finite JSON value')
def read(path):
    key=str(path.relative_to(root))
    if not path.exists():
        metadata[key]={'state':'missing'}
        return None
    try:
        stat=path.stat()
        metadata[key]={'bytes':stat.st_size,'mtime':stat.st_mtime}
        if stat.st_size>limit:
            metadata[key]['state']='omitted_size_limit'
            return None
        content=path.read_bytes()
        metadata[key]['sha256']=hashlib.sha256(content).hexdigest()
        value=json.loads(content,parse_constant=reject_nonfinite)
        metadata[key]['state']='read'
        return value
    except (OSError,ValueError) as exc:
        metadata[key]={'state':'read_error','error_type':type(exc).__name__}
        return None
result={'time':time.time(),'root':str(root),'root_exists':root.is_dir(),
        'read_only':True,'maximum_json_bytes':limit}
guide=Path('/etc/vast-agents-guide.md')
if guide.exists():
    content=guide.read_bytes()
    result['server_guide']={'path':str(guide),'sha256':hashlib.sha256(content).hexdigest(),
                            'text':content.decode('utf-8')}
else:
    result['server_guide']={'path':str(guide),'missing':True}
for name in ('manifest','registration','status','failure','result','wandb-sync-status'):
    result[name]=read(root/(name+'.json'))
result['campaign_proofs']={}
for name in ('proofs','stages'):
    for path in sorted((root/name).glob('*.json')):
        result['campaign_proofs'][str(path.relative_to(root))]=read(path)
ids={entry['id'] for entry in (result.get('manifest') or {}).get('jobs',[])}
ids.update(path.stem for path in (root/'jobs').glob('*.json'))
result['jobs']={}
names=('config','progress','result','failure','checkpoint-verification','wandb',
       'optiq-profile-verification','dacer-disabled-verification','dynamics-verification',
       'parameter-audit')
for key in sorted(ids):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',key):
        raise ValueError('unsafe job identifier in manifest')
    job={'job':read(root/'jobs'/(key+'.json'))}
    for phase in ('preflight','runs'):
        directory=root/phase/key
        files={name:read(directory/(name+'.json')) for name in names}
        extra={}
        patterns=('*proof*.json','*verification*.json','proofs/*.json','stages/*.json',
                  'policy-checkpoints/*/verification.json',
                  'replay-diagnostics/*/summary.json',
                  'evaluations/*/*/critic-diagnostics.json')
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                relative=str(path.relative_to(directory))
                if relative[:-5] not in files:
                    extra[relative]=read(path)
        files['proofs_and_diagnostics']=extra
        job[phase]=files
    result['jobs'][key]=job
result['pid_alive']={}
for entry in (result.get('status') or {}).get('running',[]):
    if isinstance(entry,dict) and entry.get('pid') is not None:
        pid=str(entry['pid'])
        if pid.isdigit():
            result['pid_alive'][pid]=Path('/proc',pid).exists()
result['file_metadata']=metadata
print(json.dumps(result,allow_nan=False))
'''.replace('__ROOT__', repr(ROOT))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(host, data):
    status = data.get('status') or {}
    rows = []
    for identifier, job in data.get('jobs', {}).items():
        run, preflight = job['runs'], job['preflight']
        progress = run.get('progress') or preflight.get('progress') or {}
        state = job.get('job') or {}
        rows.append(dict(
            id=identifier, phase=state.get('phase'), status=state.get('status'),
            steps=progress.get('steps', progress.get('step')),
            updates=progress.get('updates'), actor_updates=progress.get('actor_updates'),
            preflight_complete=bool((preflight.get('result') or {}).get('completed')),
            main_started=bool(run.get('config')),
            main_complete=bool((run.get('result') or {}).get('completed')),
            failure=run.get('failure') or preflight.get('failure'), wandb=run.get('wandb')))
    return dict(host=host, remote_time=data.get('time'), root_exists=data['root_exists'],
                registered=bool(data.get('registration')), source_commit=(data.get('manifest') or {}).get('source_commit'),
                status_present=data.get('status') is not None,
                pending=len(status.get('pending', [])) if data.get('status') is not None else None,
                running=len(status.get('running', [])) if data.get('status') is not None else None,
                completed=len(status.get('completed', [])) if data.get('status') is not None else None,
                failed=status.get('failed'), failure=data.get('failure'),
                pid_alive=data.get('pid_alive', {}), jobs=rows)


def collect(host, timeout):
    if host not in HOSTS:
        raise ValueError('Only the two approved 5090 hosts are allowed')
    process = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3', '-'],
        input=REMOTE_CODE, text=True, capture_output=True, timeout=timeout, check=True)
    return json.loads(process.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--host', action='append', choices=HOSTS,
                        help='Default: both approved hosts; may be repeated')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args(argv)
    hosts = tuple(dict.fromkeys(args.host or HOSTS))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    snapshot = args.output / 'status' / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        pending = {pool.submit(collect, host, args.timeout): host for host in hosts}
        for future in concurrent.futures.as_completed(pending):
            host = pending[future]
            try:
                data = future.result()
                atomic_json(snapshot / (host + '.json'), data)
                atomic_json(args.output / (host + '.json'), data)
                rows.append(summarize(host, data))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # Do not print arbitrary remote stderr/command lines (credentials
                # could be embedded by unrelated host startup scripts).
                error = dict(host=host, error_type=type(exc).__name__,
                             returncode=getattr(exc, 'returncode', None))
                errors.append(error)
                atomic_json(snapshot / (host + '-error.json'), error)
    result = dict(collected=time.time(), timestamp_utc=stamp, campaign=CAMPAIGN,
                  snapshot=str(snapshot.resolve()), hosts=sorted(rows, key=lambda x:x['host']),
                  errors=errors, read_only=True, complete_snapshot=len(rows)==len(hosts))
    atomic_json(snapshot / 'summary.json', result)
    atomic_json(args.output / 'latest-status.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
