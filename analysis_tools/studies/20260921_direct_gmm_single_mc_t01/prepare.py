"""Copy a verified SingleQ snapshot; change only the temperature config."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
BASE_COMMIT = '9ac88463efc44cf2a084458c322f6f6ae130d6f9'


def build(base, out, commit):
    launch = json.loads((base/'DEPLOYMENT.json').read_text())
    assert launch['commit'] == BASE_COMMIT
    assert not out.exists()
    out.mkdir(parents=True)
    for rel, digest in launch['files'].items():
        p = base/rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
        if rel.startswith('repo/') and rel != 'repo/SOURCE_MANIFEST.json':
            dest = out/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    for p in HERE.iterdir():
        if p.is_file():
            shutil.copy2(p, out/p.name)
    shutil.copy2(HERE/'config.yaml', out/'repo/configs/mujoco_direct_gmm_single_mc.yaml')
    repo = out/'repo'
    numerical = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(repo.rglob('*')) if p.is_file()}
    for rel, digest in numerical.items():
        if rel != 'configs/mujoco_direct_gmm_single_mc.yaml':
            assert digest == launch['files']['repo/'+rel], rel
    (repo/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(base_commit=BASE_COMMIT, files=numerical), indent=2)+'\n')
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'DEPLOYMENT.json').write_text(json.dumps(dict(commit=commit, branch='heejoon',
        numerical_base_commit=BASE_COMMIT, changed_numerical_files=[],
        changed_config='configs/mujoco_direct_gmm_single_mc.yaml', files=files), indent=2)+'\n')
    print('Verified snapshot files:', len(files))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    build(a.base, a.out, a.commit)
