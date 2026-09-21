"""Run on dildata using a campaign-specific read-only rrsync SSH key."""
from pathlib import Path
import fcntl
import shlex
import subprocess
import time
import traceback
from common import read, write, sha

OPS = Path('/data1/heejoonorm/OptiQ/ops/login4/single-mc-n64-m256-20260921')
OUT = Path('/data1/heejoonorm/OptiQ/studies/20260921_direct_gmm_single_mc_64x256_login4')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
       '-o', 'UserKnownHostsFile=' + str(OPS / 'known_hosts'), '-i', str(OPS / 'backup_key')]


def once():
    subprocess.run(['rsync', '-rtz', '--timeout=120', '--exclude=__pycache__', '--exclude=.pytest_cache',
                    '--exclude=*.tmp', '-e', shlex.join(SSH), 'hobbit9882@165.132.142.208:./', str(OUT) + '/'],
                   check=True, timeout=1800)
    verified = []
    for marker in OUT.glob('outputs/*/*/ARTIFACTS_SHA256.json'):
        rec = marker.parent / 'BACKUP_VERIFIED.json'
        if rec.exists():
            verified.append(str(marker.parent.relative_to(OUT)))
            continue
        files = read(marker)
        if all((marker.parent / f).exists() and sha(marker.parent / f) == h for f, h in files.items()):
            write(rec, dict(verified=True, files=len(files), time=time.time()))
            verified.append(str(marker.parent.relative_to(OUT)))
    status = dict(ok=True, time=time.time(), verified_runs=verified, total_runs=5, destination=str(OUT))
    write(OPS / 'SYNC_STATUS.json', status)
    write(OUT / 'STORAGE_SYNC_STATUS.json', status)
    return len(verified) == 5


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OPS / 'sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            if once():
                break
        except Exception:
            write(OPS / 'SYNC_ERROR.json', dict(error=traceback.format_exc(), time=time.time()))
        time.sleep(180)
