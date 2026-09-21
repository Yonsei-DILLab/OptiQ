"""Environment loading, run provenance, and scalar W&B logging."""

import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from dotenv import load_dotenv
import jax
import numpy as np
from stable_baselines3.common.logger import KVWriter


ROOT = Path(__file__).resolve().parents[1]


def load_environment():
    """Explicit dotenv, then repo dotenv, then workspace defaults; shell wins."""
    explicit = os.environ.get("OPTIQ_ENV_FILE")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError("OPTIQ_ENV_FILE does not exist")
        load_dotenv(explicit, override=False)
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT.parent / ".env", override=False)


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    versions = {}
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "optax", "numpy", "torch",
                 "gymnasium", "mujoco", "myosuite", "dm-control", "stable-baselines3",
                 "wandb", "nvidia-cudnn-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    snapshot = ROOT / "DIRECT_GMM_SOURCE.json"
    if snapshot.exists():
        source = json.loads(snapshot.read_text())
        git_info = dict(git_commit=source["base_commit"], git_dirty=True,
                        git_branch="v5 + recorded Direct GMM patch", source_manifest=source)
    else:
        git_info = dict(git_commit=git("rev-parse", "HEAD"),
                        git_dirty=bool(git("status", "--porcelain")),
                        git_branch=git("branch", "--show-current"))
    return {
        **git_info,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "packages": versions,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "gpu_models": [d.device_kind for d in jax.devices() if d.platform == "gpu"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


class WandbWriter(KVWriter):
    """Log scalars directly; avoid WandbCallback copying model.__dict__ to config."""
    def __init__(self, run):
        self.run = run

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for name, value in key_values.items():
            if isinstance(value, (int, float, np.number)):
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite metric: {name}")
                metrics[name] = scalar
        if metrics:
            # Environment steps are explicit; W&B's internal step stays monotonic
            # even if evaluation and training flush at the same environment step.
            self.run.log({"env_steps": step, **metrics})

    def close(self):
        pass
