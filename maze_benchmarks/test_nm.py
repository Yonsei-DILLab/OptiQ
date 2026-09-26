"""Sampler regression and finite-gradient checks, no training run."""
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .agents import OptiQ


def main():
    with tempfile.TemporaryDirectory() as folder:
        OptiQ(0, Path(folder), 8448, 4, 2, 5.)
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    from optiq_dime.box_gaussian import mixture_log_prob
    key = jax.random.PRNGKey(42)
    for n, m in [(64,64)] + [(n,64) for n in (1,4,16,128,256)] + [(64,m) for m in (1,4,16,128,256)]:
        mu = jax.random.uniform(key,(2,n,2),minval=-.8,maxval=.8)
        ls = jnp.full_like(mu,-1.)
        q = ConditionalGaussianProposal(mu,ls,float(np.exp(-5)))
        actions, u, ids = q.sample(key,1,'exact',count=m)
        assert actions.shape == (2,m,2) and ids.shape == (2,m)
        assert np.all(np.asarray(ids)<n) and np.all(np.abs(np.asarray(actions))<=1)
        assert np.isfinite(np.asarray(q.log_prob(u))).all()
        def loss(means, logs):
            return -mixture_log_prob(jax.lax.stop_gradient(actions),means,logs).mean()
        grads = jax.grad(loss,argnums=(0,1))(mu,ls)
        assert all(np.isfinite(np.asarray(g)).all() for g in grads)
        if n == m == 64:
            old = q.sample(key,1,'exact')
            assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(old,(actions,u,ids)))
        print(dict(N=n,M=m,finite=True),flush=True)


if __name__ == '__main__':
    main()
