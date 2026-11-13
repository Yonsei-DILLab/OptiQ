"""Local SSH relay for only the explicitly authorized env64/UTD1 cohort."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from . import archive

archive.SERVERS = [('175.155.64.205', 19159)]
archive.REMOTE_ROOT = '/workspace/heejoon-adroit-ibolt-e64u1-20261113'
archive.STORAGE_ROOT = '/data1/heejoonorm/OptiQ/runs/adroit-ibolt-e64u1-20261113'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--once', action='store_true'); a = p.parse_args()
    while True:
        try:
            reports = archive.archive_once(a.state_dir)
            print(json.dumps(dict(time=time.time(), transfers=reports)), flush=True)
            if a.once:
                break
            command = ('python3 -c ' + shlex.quote(
                'import json; from pathlib import Path; r=Path(' + repr(archive.REMOTE_ROOT) + '); '
                'q=json.loads((r/"queue-state.json").read_text()); p=r/"recovery-queue-state.json"; '
                'q["jobs"].update(json.loads(p.read_text())["jobs"] if p.exists() else {}); '
                'print(json.dumps(q))'))
            raw = subprocess.check_output([*archive.SSH, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-p', '19159', 'heejoonorm@175.155.64.205', command], text=True)
            jobs = json.loads(raw)['jobs']
            if len(jobs) == 12 and all(s['status'] in ('completed', 'failed', 'requires_review') for s in jobs.values()):
                for _ in range(2):
                    time.sleep(60)
                print(json.dumps(dict(final=True, time=time.time(),
                    transfers=archive.archive_once(a.state_dir))), flush=True)
                break
        except Exception as error:
            print(json.dumps(dict(time=time.time(), error=repr(error))), flush=True)
            if a.once:
                raise
        for _ in range(5):
            time.sleep(60)


if __name__ == '__main__':
    main()
