"""Bit-exact device arithmetic check; no RL learning or benchmark samples."""
import argparse
import json
from pathlib import Path
import time


def main(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from .run import atomic, source_provenance
    proof = source_provenance()
    assert jax.default_backend() == 'gpu' and torch.cuda.is_available()
    x = np.random.default_rng(404).integers(0, 2**24, size=2**20, dtype=np.int32)
    dx = jax.device_put(x)
    tx = torch.from_numpy(x).to('cuda')
    @jax.jit
    def transform(x, key):
        return ((x ^ key) + 12345) ^ (x >> 3)
    errors = []
    for k in range(100):
        expected = ((x ^ np.int32(k * 7919)) + 12345) ^ (x >> 3)
        a = np.asarray(transform(dx, np.int32(k * 7919)))
        b = (((tx ^ (k * 7919)) + 12345) ^ (tx >> 3)).cpu().numpy()
        for framework, value in [('jax', a), ('torch', b)]:
            count = int(np.count_nonzero(value != expected))
            if count:
                errors.append(dict(iteration=k, framework=framework, mismatches=count))
    atomic(output, dict(source_commit=proof['commit'], passed=not errors,
                        iterations=100, elements=2**20, errors=errors, time=time.time()))
    print(json.dumps(dict(passed=not errors, errors=errors)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
