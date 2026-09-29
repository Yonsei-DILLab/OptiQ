"""Read-only status collection; optionally download and verify the final archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HOST = 'vast1'
ROOT = '/home/heechan/optiq-experiments/gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925'
DEST = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-completed', action='store_true')
    args = parser.parse_args()
    code = f'''
import json,pathlib,time
r=pathlib.Path({ROOT!r})
paths=[r/n for n in ['manifest.json','registration.json','status.json','failure.json','result.json','archive.json']]
paths += list((r/'jobs').glob('*.json')) + list((r/'proofs').glob('*.json'))
for run in (r/'results').glob('ibolt_*'):
 paths += [run/n for n in ['config.json','status.json','latest.json','wandb_status.json','update_count_audit.json']]
print(json.dumps({{'collected':time.time(),'files':{{str(p.relative_to(r)):json.loads(p.read_text()) for p in paths if p.exists()}}}}))
'''
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST,
                                   'python3 -c '+shlex.quote(code)], text=True)
    data = json.loads(raw)
    for relative, value in data['files'].items():
        path = DEST/'snapshot'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+'\n')
    (DEST/'latest-snapshot.json').write_text(json.dumps(data, indent=2)+'\n')
    status = data['files']['status.json']
    print(json.dumps({key: status.get(key) for key in ('phase','running','queued','completed','failed')}, indent=2))
    if args.archive_completed and status['phase'] == 'completed':
        if 'archive.json' not in data['files']:
            print('Training/report complete; final archive is still being packaged.')
            return
        expected = data['files']['archive.json']
        local = DEST/expected['file']
        if not local.exists():
            subprocess.run(['scp', '-q', HOST+':'+ROOT+'/'+expected['file'], str(local)], check=True)
        with local.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == expected['sha256'] and local.stat().st_size == expected['bytes']
        with tarfile.open(local) as archive:
            archive.extractall(DEST/'results', filter='data')
        (DEST/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(sha256=digest, **{'bytes':local.stat().st_size}), indent=2)+'\n')


if __name__ == '__main__':
    main()
