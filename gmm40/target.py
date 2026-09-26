"""Fixed seed-0 GMM40: physical coordinates in [-40,40]^2."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
SCALE = 40.0

class Target:
    def __init__(self, definition=None):
        self.metadata = json.loads(Path(definition or Path(__file__).with_name("target.json")).read_text())
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

    def jax_log_prob(self, x):
        import jax.numpy as jnp
        import jax.scipy as jsp
        means, std = jnp.asarray(self.means), jnp.asarray(self.std)
        component = -0.5*jnp.sum(((x[...,None,:]-means)/std[:,None])**2,axis=-1)-2*jnp.log(std)-math.log(2*math.pi)
        return jsp.special.logsumexp(component,axis=-1)-math.log(40)
