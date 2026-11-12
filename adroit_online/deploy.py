"""Export committed inputs, manifest them, and transfer an immutable source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATHS = ['adroit_online', 'baselines/jax_sac.py', 'adroit_rebuttal/requirements.txt',
         'rebuttal_tools/check_fetch_jax_sac.py']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True); p.add_argument('--port', type=int, required=True)
    p.add_argument('--root', default='/workspace/heejoon-adroit-online-20261112')
    a = p.parse_args()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    commit = git('rev-parse', 'HEAD')
    assert git('branch', '--show-current') == 'heejoon-rebuttal'
    assert not git('status', '--porcelain', '--', *PATHS), 'Commit experiment inputs before deployment'
    with tempfile.TemporaryDirectory(prefix='adroit-online-deploy-') as temp:
        temp = Path(temp)
        archive = temp / 'source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '-o', str(archive), commit, *PATHS],
                       cwd=ROOT, check=True)
        source = temp / 'source'; source.mkdir()
        # The archive was generated locally from these committed repository paths.
        # System tar also works on the deployment laptop's Python3.9 runtime.
        subprocess.run(['tar', '-xf', str(archive), '-C', str(source)], check=True)
        files = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(source.rglob('*')) if f.is_file()}
        provenance = dict(commit=commit, branch='heejoon-rebuttal', files=files)
        (source / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        bundle = temp / 'source.tgz'
        with tarfile.open(bundle, 'w:gz') as t:
            for f in sorted(source.rglob('*')):
                if f.is_file(): t.add(f, arcname=str(f.relative_to(source)))
        ssh = ['ssh', '-p', str(a.port), '-o', 'BatchMode=yes', a.host]
        destination = f'{a.root}/source-{commit}'
        subprocess.run([*ssh, f'mkdir -p {shlex.quote(a.root)}/setup'], check=True)
        subprocess.run(['scp', '-P', str(a.port), '-o', 'BatchMode=yes', str(bundle),
                        f'{a.host}:{a.root}/setup/source-{commit}.tgz'], check=True)
        script = (f'test ! -e {shlex.quote(destination)} && mkdir {shlex.quote(destination)} && '
                  f'tar -xzf {shlex.quote(a.root)}/setup/source-{commit}.tgz -C {shlex.quote(destination)}')
        subprocess.run([*ssh, script], check=True)
        print(json.dumps(dict(source_commit=commit, host=a.host, port=a.port, source=destination,
                              bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
