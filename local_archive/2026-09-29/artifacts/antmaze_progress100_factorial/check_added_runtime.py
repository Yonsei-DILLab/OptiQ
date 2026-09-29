"""Read-only validation of the added host's isolated runtime and four GPUs."""
import json,os,subprocess,sys,time
from importlib import metadata
from pathlib import Path
root=Path('/home/heechan/OptiQ-ops/reviews')
expected=json.loads((root/'control-runtime.json').read_text())
actual={n:metadata.version(n) for n in expected}
assert actual==expected,{n:(expected[n],actual[n]) for n in expected if expected[n]!=actual[n]}
proof={'time':time.time(),'python':sys.version,'versions':actual,'gpu_checks':[]}
code='''import json,torch,jax,jax.numpy as jnp
x=torch.ones((32,32),device='cuda'); assert torch.isfinite(x@x).all()
y=jax.jit(lambda a:a@a)(jnp.ones((32,32)));y.block_until_ready();assert bool(jnp.isfinite(y).all())
assert jax.devices()[0].platform=='gpu'
print(json.dumps({'torch_device':torch.cuda.get_device_name(0),'jax_devices':[str(d) for d in jax.devices()]}))'''
for gpu in range(4):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda,cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,check=True)
 proof['gpu_checks'].append({'gpu':gpu,**json.loads(p.stdout)})
(root/'progress100-runtime-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
