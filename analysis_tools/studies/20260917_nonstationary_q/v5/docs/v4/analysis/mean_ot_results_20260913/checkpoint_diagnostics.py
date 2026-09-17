"""Read-only actor evaluation on a shared fixed state/latent bank; no env rollout."""
from pathlib import Path
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
import csv
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
CAMPAIGN=Path('/root/optiq-experiments/ant_v4_mean_ot_T025_20260912T235059Z')
sys.path.insert(0,str(CAMPAIGN/'source'))
import jax
import jax.numpy as jnp
from flax import serialization
from optiq_dime.policy import SemiImplicitActor

manifest=json.loads((CAMPAIGN/'manifest.json').read_text())
state=json.loads((CAMPAIGN/'state.json').read_text())
groups={'baseline':manifest['baseline_records'],'meanOT':list(state['jobs'].values())}
hashes={}
def content(path):
    b=path.read_bytes();hashes[str(path)]=hashlib.sha256(b).hexdigest();return b

bank=[]
for records in groups.values():
    for rec in records:
        file=next(Path(rec['output']).rglob('landscape_probe_batch.npz'))
        content(file)
        with np.load(file) as data:
            key='observations' if 'observations' in data else 'obs'
            bank.append(data[key][:16])
obs=np.concatenate(bank).astype(np.float32)
assert obs.shape==(128,27)
z=np.random.default_rng(9317).normal(size=(128,128,8)).astype(np.float32)
np.savez_compressed(ROOT/'fixed_probe.npz',observations=obs,latents=z)
model=SemiImplicitActor(action_dim=8,hidden_dims=(256,256),log_std_min=-5.,log_std_max=1.,initial_log_std=np.log(.5))
@jax.jit
def apply(params, obs, z):
    mu,ls=model.apply({'params':params},jnp.repeat(obs,128,axis=0),z.reshape(-1,8))
    return mu.reshape(-1,128,8),jnp.exp(ls).reshape(-1,128,8)

rows=[]
per_state=[]
for label,records in groups.items():
    for seed,rec in enumerate(records):
        for step in [5001,50000,100000,250000,1000000]:
            file=next(Path(rec['output']).rglob(f'actor_state_{step}.msgpack'))
            restored=serialization.msgpack_restore(content(file))
            params=jax.tree_util.tree_map(jnp.asarray,restored['params'])
            outputs=[apply(params,jnp.asarray(obs[k:k+16]),jnp.asarray(z[k:k+16])) for k in range(0,len(obs),16)]
            mu=np.concatenate([np.asarray(o[0]) for o in outputs])
            sigma=np.concatenate([np.asarray(o[1]) for o in outputs])
            assert np.isfinite(mu).all() and np.isfinite(sigma).all()
            meanvar=mu.var(axis=1).sum(axis=-1)
            within=(sigma**2).mean(axis=1).sum(axis=-1)
            spread=np.sqrt(np.tanh(mu).var(axis=1).sum(axis=-1))
            frac=meanvar/(meanvar+within)
            row={'variant':label,'seed':seed,'step':step,
                'mean_action_spread':float(spread.mean()), 'pretanh_mean_variance':float(meanvar.mean()),
                'conditional_noise_variance':float(within.mean()),'mean_variance_fraction':float(frac.mean()),
                'sigma_mean':float(sigma.mean()),'sigma_latent_variance':float(sigma.var(axis=1).sum(axis=-1).mean())}
            rows.append(row)
            for i in range(len(obs)):
                per_state.append({'variant':label,'seed':seed,'step':step,'state':i,
                    'mean_action_spread':float(spread[i]),'pretanh_mean_variance':float(meanvar[i]),
                    'mean_variance_fraction':float(frac[i]),'sigma_mean':float(sigma[i].mean())})
            print(json.dumps(row),flush=True)
for name,records in [('checkpoint_diagnostics.csv',rows),('checkpoint_diagnostics_per_state.csv',per_state)]:
    with (ROOT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
(ROOT/'checkpoint_verification.json').write_text(json.dumps({'passed':True,'source':str(CAMPAIGN/'source'),
    'actor_apply_file_sha256':hashlib.sha256((CAMPAIGN/'source/optiq_dime/policy.py').read_bytes()).hexdigest(),
    'states':128,'latents_per_state':128,'checkpoints':len(rows),
    'state_bank':'16 saved 50K probe observations from each of 8 runs; common across all checkpoints',
    'limitation':'Shared early-state diagnostic; not a late on-policy state distribution or a marginal density fit metric',
    'checkpoint_or_source_writes':False,'input_hashes':hashes},indent=2)+'\n')
