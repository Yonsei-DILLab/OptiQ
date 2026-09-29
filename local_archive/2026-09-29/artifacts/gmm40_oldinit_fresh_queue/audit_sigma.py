"""Read-only final-checkpoint forward pass on CPU; no training or queue changes."""
import json
from pathlib import Path
import subprocess

CODE=r'''
import json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax.serialization import msgpack_restore
from gmm40.optiq_trg import Actor
root=Path('/home/heechan/optiq-experiments/gmm40-trg-oldinit-fresh-100k-4seed-20260921/results')
rows=[]
for seed in range(4):
 folder=root/f'oldinit_fresh_optiq_trg_s{seed}_100k'
 cfg=json.loads((folder/'config.json').read_text());saved=msgpack_restore((folder/'checkpoints/step_0100000.bin').read_bytes())
 assert saved['updates']==saved['state']['step']==100000
 p=saved['state']['params'];key,_=jax.random.split(jax.random.PRNGKey(900000+seed))
 z=jax.random.normal(key,(10000,2));obs=jnp.zeros((len(z),1));h=jnp.concatenate([obs,z],axis=-1)
 for layer in ['Dense_0','Dense_1']:h=jax.nn.gelu(h@p[layer]['kernel']+p[layer]['bias'])
 raw=np.asarray(h@p['log_std']['kernel']+p['log_std']['bias'])
 actor=Actor(2,(256,256),-5.,1.,cfg['initial_log_std'],mean_output_init_scale=1.)
 mu,ls=actor.apply({'params':p},obs,z)
 np.testing.assert_allclose(ls,np.clip(raw,-5,1),atol=1e-6)
 saved_mu=np.load(folder/'evaluations/step_0100000/samples_mu_only.npy')
 rows.append(dict(seed=seed,checkpoint_updates=int(saved['updates']),diagnostic_latents=10000,
  raw_log_sigma_min=float(raw.min()),raw_log_sigma_max=float(raw.max()),
  raw_log_sigma_quantiles=np.quantile(raw,[0,.25,.5,.75,.9,.99,1]).tolist(),
  coordinate_fraction_above_cap=float((raw>1).mean()),latent_fraction_any_coordinate_above_cap=float((raw>1).any(-1).mean()),
  sigma_parameter_max=float(np.exp(ls).max()),sigma_parameter_mean=float(np.exp(ls).mean()),
  stored_mu_vs_cpu_max_error=float(np.abs(saved_mu-40*np.array(mu)).max())))
print(json.dumps(dict(source_commit=cfg['source_git_commit'],backend=jax.default_backend(),
 cap_log_sigma=1.,cap_sigma_parameter=float(np.exp(1)),coordinate_system='normalized a; physical x=40*a; parameter sigma is not the truncated-distribution standard deviation',
 rows=rows)))
'''
cmd='cd /home/heechan/OptiQ-ops/sources/85aee3ecae09847954cb999680f854302f148db6 && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/heechan/.venv-optiq-gmm40/bin/python -'
p=subprocess.run(['ssh','-o','ConnectTimeout=15','vast-heechan-180',cmd],input=CODE,text=True,capture_output=True,timeout=120)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
data=json.loads(p.stdout);out=Path(__file__).resolve().parent/'diagnosis';out.mkdir(exist_ok=True)
(out/'sigma_checkpoint_audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
