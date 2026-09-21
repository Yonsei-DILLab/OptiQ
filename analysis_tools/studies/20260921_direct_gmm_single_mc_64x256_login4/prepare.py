"""Build from verified SingleQ MC64; decouple teacher count from N and K."""
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
    p = out/'repo/run_optiq_dime.py'
    text = p.read_text()
    old = '                and actor.proposals_per_policy_sample == 1):'
    assert text.count(old) == 1
    text = text.replace(old, '                and isinstance(actor.proposals_per_policy_sample, int)\n'
                             '                and actor.proposals_per_policy_sample >= 1):')
    text = text.replace('This campaign requires single scalar critic and N=M=K',
                        'This campaign requires single scalar critic, N=K and a positive integer candidate multiplier')
    p.write_text(text)
    p = out/'repo/configs/mujoco_direct_gmm_single_mc.yaml'
    text = p.read_text().replace('proposals_per_policy_sample: 1', 'proposals_per_policy_sample: 4')
    text = text.replace('M64', 'M256').replace('20260921_DirectGMM_SingleQ_MC64_N64_M256_T025',
        '20260921_DirectGMM_SingleQ_MC64_N64_M256_T025_login4')
    p.write_text(text)
    files = {str(p.relative_to(out/'repo')): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((out/'repo').rglob('*')) if p.is_file()}
    changed = [r for r,h in files.items() if h != launch['files']['repo/'+r]]
    assert set(changed) == {'run_optiq_dime.py', 'configs/mujoco_direct_gmm_single_mc.yaml'}, changed
    (out/'repo/SOURCE_MANIFEST.json').write_text(json.dumps(dict(base_commit=BASE_COMMIT,
        changed_files=changed, files=files),indent=2)+'\n')
    full = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'DEPLOYMENT.json').write_text(json.dumps(dict(commit=commit, branch='heejoon',
        numerical_base_commit=BASE_COMMIT, files=full),indent=2)+'\n')
    print('Verified build:',len(full),'files; only configuration and entry-point validation changed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    build(a.base,a.out,a.commit)
