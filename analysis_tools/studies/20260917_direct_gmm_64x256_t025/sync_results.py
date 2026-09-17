"""Independent dildata collector for the queued campaign; no remote credentials copied."""
from pathlib import Path
import fcntl
import json
import shlex
import subprocess
import time
import traceback
from controller import write, read, sha

OPS = Path('/data1/heejoonorm/OptiQ/ops/vast/51277568')
STATE = OPS / 'N64_M256_T025'
OUT = Path('/data1/heejoonorm/OptiQ/runs/vast-51277568/DirectGMM_N64_M256_T025')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
       '-o', 'UserKnownHostsFile=' + str(OPS / 'known_hosts'),
       '-i', str(OPS / 'backup_key'), '-p', '35221']


def once():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-rtz', '--exclude=__pycache__', '--exclude=.pytest_cache',
                    '--exclude=controller.lock', '-e', shlex.join(SSH),
                    'root@45.143.122.5:campaigns/20260917_N64_M256_T025/', str(OUT) + '/'],
                   check=True, timeout=600)
    verified = []
    for marker in OUT.glob('outputs/*/*/ARTIFACTS_SHA256.json'):
        rec = marker.parent / 'BACKUP_VERIFIED.json'
        if rec.exists():
            verified.append(str(marker.parent.relative_to(OUT)))
            continue
        files = read(marker)
        if all((marker.parent / p).exists() and sha(marker.parent / p) == h for p, h in files.items()):
            write(rec, dict(verified=True, files=len(files), time=time.time()))
            verified.append(str(marker.parent.relative_to(OUT)))
    write(STATE / 'SYNC_STATUS.json', dict(ok=True, time=time.time(),
          verified_runs=verified, destination=str(OUT)))
    return len(verified) == 20 and (OUT / 'ALL_COMPLETE.json').exists()


if __name__ == '__main__':
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            if once():
                break
        except Exception:
            write(STATE / 'SYNC_ERROR.json', dict(error=traceback.format_exc(), time=time.time()))
        time.sleep(120)
