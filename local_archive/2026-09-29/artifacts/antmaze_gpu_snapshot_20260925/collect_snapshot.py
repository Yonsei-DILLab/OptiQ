"""Archive the latest complete evaluation of each live 5090 AntMaze run."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent
HOSTS = ("vast-heechan-180", "vast-heechan-199")

REMOTE = r'''
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile

host = sys.argv[1]
base = Path('/home/heechan/optiq-experiments')
names = (
    'antmaze-optiq-utd1-random-reward-pair-v34-s0-20260925',
    'antmaze-optiq-utd1-random-progress100-baseline-replacement-v34-s0-20260925',
    'antmaze-optiq-fixed-start-nearest-dense-gamma09-v34-1m-s0-20260925',
)
archive = Path('/tmp/antmaze-gpu-snapshot-20260925-' + host + '.tar.gz')
index = {'host': host, 'runs': [], 'files': {}}
with tarfile.open(archive, 'w:gz') as tar:
    def add(path):
        relative = Path(host) / path.relative_to(base)
        if not path.is_file():
            raise FileNotFoundError(path)
        tar.add(path, arcname=str(relative), recursive=False)
        index['files'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for name in names:
        campaign = base / name
        status = json.loads((campaign / 'status.json').read_text())
        add(campaign / 'manifest.json')
        add(campaign / 'status.json')
        for job in status['running']:
            run = campaign / 'runs' / job['id']
            config_path = run / 'config.json'
            progress_path = run / 'progress.json'
            if config_path.exists():
                add(config_path)
            if progress_path.exists():
                add(progress_path)
            selected = None
            if (run / 'evaluations').exists():
                for evaluation in sorted((run / 'evaluations').iterdir(), reverse=True):
                    if not evaluation.is_dir():
                        continue
                    modes = [p for p in evaluation.iterdir() if p.is_dir()
                             and (p / 'summary.json').exists()
                             and (p / 'rollouts.npz').exists()]
                    if any(p.name.startswith('native-') for p in modes) and any(
                            p.name.startswith('policy-') for p in modes):
                        selected = evaluation
                        break
            modes = []
            if selected:
                for mode in selected.iterdir():
                    if not mode.is_dir() or not (mode / 'summary.json').exists() or not (mode / 'rollouts.npz').exists():
                        continue
                    add(mode / 'summary.json')
                    add(mode / 'rollouts.npz')
                    modes.append(mode.name)
            index['runs'].append(dict(campaign=name, job=job['id'], gpu=job['gpu'],
                                      eval_step=int(selected.name) if selected else None,
                                      modes=sorted(modes)))
    data = (json.dumps(index, indent=2) + '\n').encode()
    info = tarfile.TarInfo(host + '/snapshot-index.json')
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps(dict(archive=str(archive), sha256=sha, index=index)))
'''


def collect():
    ROOT.mkdir(parents=True, exist_ok=True)
    local = []
    for host in HOSTS:
        proc = subprocess.run(["ssh", host, "python3", "-", host], input=REMOTE,
                              text=True, capture_output=True, check=True)
        record = json.loads(proc.stdout.splitlines()[-1])
        target = ROOT / f"{host}.tar.gz"
        subprocess.run(["scp", f"{host}:{record['archive']}", str(target)], check=True,
                       stdout=subprocess.DEVNULL)
        observed = hashlib.sha256(target.read_bytes()).hexdigest()
        if observed != record['sha256']:
            raise ValueError(f"SHA256 mismatch: {host}")
        with tarfile.open(target) as tar:
            for entry in tar.getmembers():
                if not entry.isfile():
                    raise ValueError(f"Unexpected archive entry: {entry.name}")
                destination = (ROOT / 'raw' / entry.name).resolve()
                if not destination.is_relative_to((ROOT / 'raw').resolve()):
                    raise ValueError(entry.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as source, destination.open('wb') as output:
                    output.write(source.read())
        for relative, sha in record['index']['files'].items():
            path = ROOT / 'raw' / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Extracted file mismatch: {relative}")
        local.append(dict(host=host, archive=str(target), sha256=observed,
                          runs=record['index']['runs'], verified_files=len(record['index']['files'])))
        print(host, [(r['gpu'], r['job'], r['eval_step']) for r in record['index']['runs']])
    (ROOT / 'collection.json').write_text(json.dumps(local, indent=2) + '\n')


if __name__ == '__main__':
    collect()
