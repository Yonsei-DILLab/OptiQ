"""SAC tanh-Gaussian actor (pre-tanh mean / std) on given iBOLT evaluation states (sbx-venv, CPU).

Usage (repo root): JAX_PLATFORMS=cpu python results/modal_em/sac_ph_extract.py <sac_run_dir> <ckpt> <out.npz> <states.npz> [stride]
"""
import json, sys
from pathlib import Path
import numpy as np
import jax
from sbx import SAC

run, ckpt, out, sfile = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
model = SAC.load(run / "checkpoints" / f"{ckpt}.zip", device="cpu")
st = model.policy.actor_state


@jax.jit
def musig(obs):
    d = st.apply_fn(st.params, obs)
    return d.distribution.mean(), d.distribution.stddev()


S = np.load(sfile)["states"][::stride].astype(np.float32)
mu, sd = map(np.asarray, musig(S))
np.savez(out, mu=mu, sigma=sd, states=S, run=run.name, ckpt=ckpt, states_from=sfile, stride=stride)
print(f"{run.name.split('_2026')[0]} {ckpt} on {sfile}[::{stride}]: {S.shape}, sigma mean {sd.mean():.3f} "
      f"(frac > 1/sqrt2: {(sd > 2 ** -0.5).mean():.3f}), |mu| mean {np.abs(mu).mean():.2f}")
