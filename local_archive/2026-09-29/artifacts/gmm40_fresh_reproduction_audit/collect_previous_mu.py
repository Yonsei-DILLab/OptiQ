"""Read-only checkpoint evaluation with fresh latent; never train or edit remote files."""
import io,json,subprocess
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent/'previous_mu';out.mkdir(exist_ok=True)
remote=r'''
import io,json,sys,subprocess
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax.serialization import msgpack_restore
from experiments.gmm_gradient_interference import bootstrap
from optiq_dime.policy import SemiImplicitActor
root=Path('/home/heechan/optiq-experiments/gmm40-fixed-fresh-b256-20260920')
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert sha=='184bd7e26736a3af136f23b028f52168a34e80e0'
model=SemiImplicitActor(2,(256,256),-5.,1.,float(np.log(.5)),mean_output_init_scale=1.)
arrays={};report=[]
for seed in [0,1]:
 p=root/f'fresh_seed{seed}';m=json.loads((p/'manifest.json').read_text())
 ck=msgpack_restore((p/'checkpoint.msgpack').read_bytes());params=ck['state']['params']
 assert m['mode']=='fresh' and int(ck['step'])==int(ck['state']['step'])==100000
 zk,ek=jax.random.split(jax.random.PRNGKey(80000+seed))
 z=jax.random.normal(zk,(32768,2))
 mu,ls=model.apply({'params':params},jnp.zeros((len(z),1)),z)
 centers=40*jnp.tanh(mu)
 full=40*jnp.tanh(mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape))
 saved=np.load(p/'step100000.npz')
 bmu,bls=model.apply({'params':params},jnp.zeros((64,1)),jnp.asarray(saved['z']))
 arrays[f'centers_s{seed}']=np.asarray(centers)
 arrays[f'full_cpu_s{seed}']=np.asarray(full)
 arrays[f'z_s{seed}']=np.asarray(z)
 arrays[f'log_sigma_s{seed}']=np.asarray(ls)
 arrays[f'full_saved_s{seed}']=saved['samples']
 arrays[f'reference_s{seed}']=np.load(p/'reference.npy')
 report.append(dict(seed=seed,source_commit=sha,checkpoint_step=100000,latent_mode='fresh',
   latent_key=80000+seed,count=len(z),evaluation_backend=jax.default_backend(),optimizer_updates=0,
   raw_mu_bank_max_cpu_gpu_difference=float(np.max(np.abs(bmu-saved['mu']))),
   full_sample_max_cpu_gpu_difference=float(np.max(np.abs(full-saved['samples']))),
   center_definition='40*tanh(mu(z)); sigma noise removed, not conditional expectation'))
arrays['metadata']=np.asarray(json.dumps(report))
buf=io.BytesIO();np.savez_compressed(buf,**arrays);sys.stdout.buffer.write(buf.getvalue())
'''
cmd='cd /home/heechan/OptiQ-heejoon && JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/heechan/.venv-optiq-gmm40/bin/python -'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','vast-heechan-199',cmd],input=remote.encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode())
a=np.load(io.BytesIO(r.stdout));(out/'evaluations.npz').write_bytes(r.stdout)
(out/'provenance.json').write_text(str(a['metadata'])+'\n')
print(str(a['metadata']))
