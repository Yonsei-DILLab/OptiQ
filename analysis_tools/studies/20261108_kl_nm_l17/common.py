import hashlib
import json
import os
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parent


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')
        f.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tasks(phase):
    base = json.loads((SOURCE / 'config.json').read_text())
    result = []
    pairs = ([(128, 128), (32, 32), (512, 512)] if phase == 'pilot'
             else [(n, m) for n in base['grid'] for m in base['grid']])
    seeds = [0] if phase == 'pilot' else base['seeds']
    for n, m in pairs:
        for seed in seeds:
            for case in base['cases']:
                for method in ['forward', 'reverse']:
                    if phase == 'pilot' and method != 'reverse':
                        continue
                    run_id = f'{case["id"]}__{method}_N{n}_M{m}_s{seed}'
                    reused = phase != 'pilot' and method == 'forward' and run_id in base['reuse_forward_ids']
                    if (phase == 'reuse') != reused and phase != 'pilot':
                        continue
                    cfg = dict(base, **case, n=n, m=m,
                               train_block=5 if method == 'reverse' else 100,
                               density_chunk=4096 if method == 'reverse' else 256,
                               checkpoint_interval=100 if method == 'reverse' else 1000)
                    result.append(dict(id=run_id,
                                       target=case['id'], method=method, n=n, m=m,
                                       L=base['reverse_L'] if method == 'reverse' else 0,
                                       seed=seed, phase=phase, config=cfg,
                                       config_sha256=digest(cfg)))
    return result


def verify_source(root):
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for rel, expected in manifest['files'].items():
        assert file_sha(SOURCE / rel) == expected, f'Source changed: {rel}'
    return manifest['commit']


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
