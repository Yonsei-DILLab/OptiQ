"""Minimal runtime metadata, without hostnames or user paths."""
import importlib.metadata
import math
from pathlib import Path
import subprocess
import jax
import numpy as np
from stable_baselines3.common.logger import KVWriter
ROOT = Path(__file__).resolve().parents[1]

def provenance():
    versions = {}
    for name in ('jax','jaxlib','flax','optax','numpy','gymnasium','mujoco','stable-baselines3','wandb'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(revision=revision, packages=versions, backend=jax.default_backend(),
                devices=[d.device_kind for d in jax.devices()])

class WandbWriter(KVWriter):
    def __init__(self, run):
        self.run = run
    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for name, value in key_values.items():
            if isinstance(value, (int,float,np.number)):
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f'Nonfinite metric: {name}')
                metrics[name] = scalar
        if metrics:
            self.run.log({'env_steps': step, **metrics})
    def close(self):
        pass
