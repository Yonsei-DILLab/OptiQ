"""Archive completed checkpoints and verify exact remote proof hashes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results/vast-heechan-180'
REMOTE='/home/heechan/optiq-experiments/antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2'
SOURCE='d25930197ee9ba060a3aa84c3b6fea419dfe856d'

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def archive(job):
    folder=RESULTS/'runs'/job['id']
    result=json.loads((folder/'result.json').read_text())
    proof=json.loads((folder/'checkpoint-verification.json').read_text())
    assert result['completed'] and result['source_commit']==SOURCE
    assert result['steps']==258304 and result['updates']==7816
    assert proof==result['checkpoint'] and proof['readback_verified']
    assert proof['environment_reward_verified'] and proof['progress_replay_verified']
    out=ROOT/'checkpoints'/job['id'];out.mkdir(parents=True,exist_ok=True)
    destination=out/'checkpoint-final.pt'
    if not destination.exists():
        assert shutil.disk_usage(out).free>4*proof['bytes']
        with tempfile.NamedTemporaryFile(dir=out,prefix='checkpoint.download.',delete=False) as handle:
            temporary=Path(handle.name)
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                'vast-heechan-180','cat',shlex.quote(REMOTE+'/runs/'+job['id']+'/checkpoint-final.pt')],
                stdout=handle,stderr=subprocess.PIPE,timeout=180)
        assert p.returncode==0, (job['id'],p.returncode)
        assert temporary.stat().st_size==proof['bytes'] and digest(temporary)==proof['sha256']
        temporary.replace(destination)
    assert destination.stat().st_size==proof['bytes'] and digest(destination)==proof['sha256']
    record=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,
        job=job['id'],local_path=str(destination),sha256=proof['sha256'],bytes=proof['bytes'],
        remote_checkpoint_proof=proof,local_bytes_verified=True,training_restarted=False,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'archive-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return {k:record[k] for k in ['job','sha256','bytes','local_bytes_verified']}

manifest=json.loads((RESULTS/'manifest.json').read_text())
assert manifest['source_commit']==SOURCE
with ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(archive,manifest['jobs']):
        print(json.dumps(result),flush=True)
