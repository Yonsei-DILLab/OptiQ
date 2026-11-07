"""Fail unless JAX and PyTorch can execute on every visible GPU."""
import jax
import jax.numpy as jnp
import numpy as np
import torch

devices = jax.devices("gpu")
assert devices, "No JAX GPU devices"
for device in devices:
    with jax.default_device(device):
        result = jax.jit(lambda x: x @ x)(jnp.ones((32, 32)))
        np.testing.assert_allclose(np.asarray(result.block_until_ready()), 32)
    print(f"JAX {jax.__version__}: {device}", flush=True)
assert torch.cuda.device_count() == len(devices), "JAX/PyTorch GPU count differs"
for index in range(torch.cuda.device_count()):
    x = torch.ones((32, 32), device=f"cuda:{index}")
    torch.testing.assert_close(x @ x, torch.full_like(x, 32))
    torch.cuda.synchronize(index)
    print(f"PyTorch {torch.__version__}: GPU {index} {torch.cuda.get_device_name(index)}", flush=True)
