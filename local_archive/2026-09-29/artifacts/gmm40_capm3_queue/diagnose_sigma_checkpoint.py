"""Read-only CPU checkpoint audit; no training, source or queue changes."""
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax.serialization import msgpack_restore
from gmm40.optiq_trg import Actor
root=Path('/home/heechan/optiq-experiments/gmm40-trg-capm3-mean1-100k-4seed-20260921/results')
target=json.loads((root/'target/definition.json').read_text())
means=np.array(target['means']);std=np.array(target['std'])
result=[]
for seed in [0,1,2,3]:
    folder=root/f'capm3_mean1_optiq_trg_s{seed}_100k'
    saved=msgpack_restore((folder/'checkpoints/step_0100000.bin').read_bytes())
    params=saved['state']['params']
    zk,_=jax.random.split(jax.random.PRNGKey(900000+seed))
    z=jax.random.normal(zk,(10000,2));obs=jnp.zeros((len(z),1))
    x=jnp.concatenate([obs,z],axis=-1)
    for layer in ['Dense_0','Dense_1']:
        x=jax.nn.gelu(x@params[layer]['kernel']+params[layer]['bias'])
    raw=np.asarray(x@params['log_std']['kernel']+params['log_std']['bias'])
    mu=np.asarray(jnp.tanh(x@params['mu']['kernel']+params['mu']['bias']))
    ls=np.clip(raw,-5,-3)
    actor=Actor(2,(256,256),-5.,-3.,-3.)
    real_mu,real_ls=actor.apply({'params':params},obs,z)
    np.testing.assert_allclose(mu,real_mu,atol=1e-6)
    np.testing.assert_allclose(ls,real_ls,atol=1e-6)
    saved_mu=np.load(folder/'evaluations/step_0100000/samples_mu_only.npy')
    mu_errors=np.abs(40*mu-saved_mu)
    max_mu_error=float(mu_errors.max())
    samples=np.load(folder/'evaluations/step_0100000/samples.npy')
    near=(((samples[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1)<=9
    cap=raw>-3;anycap=cap.any(-1)
    result.append(dict(seed=seed,checkpoint_step=int(saved['updates']),
        optimizer_step=int(saved['state']['step']),sample_count=len(samples),
        fraction_sigma_coordinates_above_cap=float(cap.mean()),
        fraction_samples_any_sigma_coordinate_above_cap=float(anycap.mean()),
        near_given_any_cap=float(near[anycap].mean()) if anycap.any() else None,near_given_no_cap=float(near[~anycap].mean()) if (~anycap).any() else None,
        outside_samples_associated_with_cap=float((anycap & ~near).sum()/(~near).sum()),
        raw_log_sigma_quantiles=np.quantile(raw,[0,.25,.5,.75,.9,1]).tolist(),
        physical_sigma_parameter_mean=float((40*np.exp(ls)).mean()),
        cap_sigma_physical_parameter=float(40*np.exp(-3)),target_sigma=float(std[0]),
        saved_mu_vs_cpu_forward_max_error=max_mu_error,
        saved_mu_vs_cpu_forward_error_quantiles=np.quantile(mu_errors,[.5,.9,.99,.999,1]).tolist(),
        association_caveat='Cap groups use CPU recomputation; saved samples use GPU. Forward differences are reported.'))
print(json.dumps(dict(runs=result,clip_derivatives={str(v):float(jax.grad(lambda x:jnp.clip(x,-5.,-3.))(jnp.float32(v)))
    for v in [-4.,-3.,-2.,0.]},source_commit='17ae649d0b265bae0d01425435172f74ae3881a6',
    interpretation='Observed cap saturation and sample association, not a controlled attribution across all changed settings.'),indent=2))
'''
command=('cd /home/heechan/OptiQ-ops/sources/17ae649d0b265bae0d01425435172f74ae3881a6 && '
         'JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '
         '/home/heechan/.venv-optiq-gmm40/bin/python -')
run=subprocess.run(['ssh','-o','ControlMaster=auto','-o','ControlPersist=600',
    '-o','ControlPath=/tmp/optiq-5090-%r-%h-%p','vast-heechan-180',command],
    input=REMOTE,text=True,capture_output=True,timeout=90)
if run.returncode:raise RuntimeError(run.stderr)
data=json.loads(run.stdout)
dest=Path(__file__).resolve().parent/'diagnosis';dest.mkdir(exist_ok=True)
(dest/'sigma_checkpoint_audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
