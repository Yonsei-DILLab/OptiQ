"""Archive completed sigmacap checkpoints; never launch or resume training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
HOST = 'vast-heechan-199'
RESULTS = ROOT / 'results' / HOST
REMOTE = '/home/heechan/optiq-experiments/antmaze-optiq-sigmacap-v34-250k-s0-20260925'
SOURCE = '6e9090c8b9bb0ed7d1700ab6339a481bc97bf84d'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def archive(job):
    folder = RESULTS / 'runs' / job['id']
    if not (folder / 'result.json').is_file():
        return {'job': job['id'], 'status': 'not_yet_completed'}
    result = json.loads((folder / 'result.json').read_text())
    proof = json.loads((folder / 'checkpoint-verification.json').read_text())
    assert result['completed'] and result['source_commit'] == SOURCE
    assert result['steps'] == 258304 and result['updates'] == 7816
    assert proof == result['checkpoint'] and proof['readback_verified']
    assert proof['environment_reward_verified'] and proof['progress_replay_verified']
    out = ROOT / 'checkpoints' / job['id']
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'checkpoint-final.pt'
    if not destination.exists():
        assert shutil.disk_usage(out).free > 4 * proof['bytes']
        # A slow SSH link can outlast a single streaming copy. Keep an explicitly
        # incomplete local file and use compressed rsync to continue its bytes.
        # The original server checkpoint is only read, never modified.
        temporary = out / 'checkpoint-final.pt.partial'
        old_partials = sorted(out.glob('checkpoint.download.*'),
                              key=lambda path: path.stat().st_size, reverse=True)
        if not temporary.exists() and old_partials:
            old_partials[0].replace(temporary)
        command = ['rsync', '--partial', '--compress', '--inplace', '--timeout=120',
                   '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                   HOST + ':' + REMOTE + '/runs/' + job['id'] + '/checkpoint-final.pt',
                   str(temporary)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        assert process.returncode == 0, (job['id'], process.returncode, process.stderr[-1000:])
        assert temporary.stat().st_size == proof['bytes'] and digest(temporary) == proof['sha256']
        temporary.replace(destination)
    assert destination.stat().st_size == proof['bytes'] and digest(destination) == proof['sha256']
    record = dict(
        time_utc=datetime.now(timezone.utc).isoformat(), source_commit=SOURCE,
        job=job['id'], local_path=str(destination), sha256=proof['sha256'], bytes=proof['bytes'],
        remote_checkpoint_proof=proof, local_bytes_verified=True, training_restarted=False,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'archive-verification.json').write_text(json.dumps(record, indent=2) + '\n')
    return {key: record[key] for key in ('job', 'sha256', 'bytes', 'local_bytes_verified')}


if __name__ == '__main__':
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    assert manifest['source_commit'] == SOURCE
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(archive, manifest['jobs']):
            print(json.dumps(result), flush=True)
