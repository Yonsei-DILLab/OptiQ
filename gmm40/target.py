"""The exact DiKL seed-0 target and its explicitly bounded counterpart.

Ground-truth sampling lives in evaluation only. Learners receive log density.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtr

ROOT = Path(os.environ.get("GMM40_REPO_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = Path(os.environ.get("GMM40_RESULTS_ROOT", ROOT / "gmm40-results"))
SCALE = 40.0


def initialize_target():
    import torch
    path = ROOT / "gmm40-baseline/DiKL/DiKL/energy/mog40.py"
    spec = importlib.util.spec_from_file_location("dikl_original_mog40", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        original = module.GMM(2, 40, 40, log_var_scaling=1, seed=0, device="cpu")
    means = original.locs.cpu().numpy()
    std = original.scale_trils[:, 0, 0].cpu().numpy()
    mass = np.prod(ndtr((SCALE-means)/std[:, None])-ndtr((-SCALE-means)/std[:, None]), axis=1)
    metadata = dict(means=means.tolist(), std=std.tolist(), weights=[1/40]*40,
                    original_source=str(path), original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    seed=0, dimension=2, scale=SCALE, bounded_mass=float(mass.mean()),
                    outside_mass=float(1-mass.mean()), component_bounded_mass=mass.tolist(),
                    Q="log p_original(x)", temperature=1.0,
                    evaluated_target="p_original(x) / Z_box inside (-40,40)^2")
    folder = RESULTS / "target"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "definition.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == metadata, "Target changed"
    else:
        destination.write_text(json.dumps(metadata, indent=2)+"\n")
    return Target()


class Target:
    def __init__(self):
        self.metadata = json.loads((RESULTS/"target/definition.json").read_text())
        self.means = np.asarray(self.metadata["means"])
        self.std = np.asarray(self.metadata["std"])
        self.log_z = math.log(self.metadata["bounded_mass"])

    def log_prob(self, x, bounded=False):
        x = np.asarray(x)
        diff = (x[..., None, :]-self.means)/self.std[:, None]
        lp = logsumexp(-0.5*np.sum(diff**2, axis=-1)-2*np.log(self.std)-math.log(2*math.pi), axis=-1)-math.log(40)
        if bounded:
            lp = np.where(np.all(np.abs(x)<SCALE, axis=-1), lp-self.log_z, -np.inf)
        return lp

    def sample(self, n, seed, bounded=True):
        rng = np.random.default_rng(seed)
        pieces, count = [], 0
        while count < n:
            indices = rng.integers(40, size=max(1024, n-count))
            x = self.means[indices]+self.std[indices,None]*rng.normal(size=(len(indices),2))
            if bounded:
                x = x[np.all(np.abs(x)<SCALE,axis=1)]
            pieces.append(x)
            count += len(x)
        return np.concatenate(pieces)[:n].astype(np.float32)

    def torch_log_prob(self, x):
        import torch
        means = torch.as_tensor(self.means, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        component = -0.5*(((x[...,None,:]-means)/std[:,None])**2).sum(-1)-2*std.log()-math.log(2*math.pi)
        return torch.logsumexp(component,dim=-1)-math.log(40)

    def jax_log_prob(self, x):
        import jax.numpy as jnp
        import jax.scipy as jsp
        means, std = jnp.asarray(self.means), jnp.asarray(self.std)
        component = -0.5*jnp.sum(((x[...,None,:]-means)/std[:,None])**2,axis=-1)-2*jnp.log(std)-math.log(2*math.pi)
        return jsp.special.logsumexp(component,axis=-1)-math.log(40)
