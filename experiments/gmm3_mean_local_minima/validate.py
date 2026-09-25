import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize._numdiff import approx_derivative
from .core import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=load_config();rows=[]
    for case in [cases(cfg)[1],cases(cfg)[-1]]:
        t=Trainer(case,cfg);mu=np.asarray(t.mu[0]);v,g=jax.value_and_grad(t.fn)(jnp.asarray(mu));h=jax.hessian(t.fn)(jnp.asarray(mu))
        va,ga,ha,err=adaptive(mu,case['centers'],cfg['sigma'])
        assert abs(float(v)-va)<1e-9 and np.max(np.abs(g-ga))<1e-9 and np.max(np.abs(h-ha))<1e-8
        fd=approx_derivative(lambda m:np.asarray(jax.grad(t.fn)(jnp.asarray(m))),mu)
        assert np.max(np.abs(fd-h))<1e-7
        init=np.asarray(t.mu);t.mu=t.advance(t.mu,100);jax.block_until_ready(t.mu)
        assert np.all(np.asarray(jax.vmap(t.fn)(t.mu))<=np.asarray(jax.vmap(t.fn)(init)))
        rows.append(dict(case=case['id'],adaptive_error_estimate=err,gradient_difference=float(np.max(np.abs(g-ga))),hessian_difference=float(np.max(np.abs(h-ha)))))
    d=dict(passed=True,checks=rows,devices=str(jax.devices()));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
