"""Create a small immutable, non-anonymous experiment source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    assert branch and 'anonymous' not in branch.lower()
    revision = 'bdbc3c989fb59de04a8e8b3700bcd61aaabbcfc1'
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision], cwd=root, text=True).splitlines()
    chosen = [f for f in files if f in ('train.py', 'environment.py', 'exploration.py')
              or f.startswith(('ibolt/', 'models/', 'configs/')) and f.endswith(('.py', '.yaml'))]
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=False)
    for name in chosen:
        target = dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', revision+':'+name], cwd=root))
    original = (dest/'models/critic.py').read_text()
    (dest/'models/native_critic.py').write_text(original)
    anchor = '        return (x - mean) * jax.lax.rsqrt(var + self.epsilon) * scale + bias'
    capture = '''        if (not self.use_running_average and not self.is_initializing()
                and self.is_mutable_collection('reference_stats')):
            entries = dict(mean=mean, var=var, steps=steps.value,
                r_clip_fraction=jnp.mean(((std/ra_std < 1/self.r_max) |
                                         (std/ra_std > self.r_max)).astype(x.dtype)),
                d_clip_fraction=jnp.mean((jnp.abs((batch_mean - old_ra_mean)/ra_std)
                                         > self.d_max).astype(x.dtype)), warm=warm)
            for name, value in entries.items():
                self.variable('reference_stats', name, lambda: value).value = value
'''
    assert original.count(anchor) == 1
    patched = original.replace('                std = jnp.sqrt(batch_var + self.epsilon)',
        '                old_ra_mean = ra_mean.value\n                std = jnp.sqrt(batch_var + self.epsilon)')
    patched = patched.replace(anchor, capture+anchor)
    patched = patched.replace("ensemble = nn.vmap(CrossQCritic,variable_axes={'params':0,'batch_stats':0}",
                              "ensemble = nn.vmap(CrossQCritic,variable_axes={'params':0,'batch_stats':0,'reference_stats':0}")
    assert patched != original and "'reference_stats':0" in patched
    (dest/'models/critic.py').write_text(patched)
    for name in ('crossq_candidate_norm.py', 'check_crossq_candidate_norm.py',
                 'diagnose_crossq_candidate_norm.py'):
        target = dest/'rebuttal_tools'/name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((root/'rebuttal_tools'/name).read_bytes())
    hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in dest.rglob('*') if p.is_file()}
    (dest/'source.json').write_text(json.dumps(dict(base_commit=revision, branch=branch,
        files=hashes, note='Candidate normalization ablation; original core files are not edited.'), indent=2))
    print(json.dumps(dict(destination=str(dest), files=len(hashes), base_commit=revision)))


if __name__ == '__main__':
    main()
