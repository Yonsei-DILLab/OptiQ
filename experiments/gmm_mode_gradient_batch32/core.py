"""32 independent same-state groups; one Adam step on their mean gradient."""
from experiments.gmm_mode_gradient.core import *

BATCH_SIZE = 32
MAX_PARALLEL_PAIRS = 8388608


def microbatch_size(n, m, batch=BATCH_SIZE):
    return max(k for k in (1, 2, 4, 8, 16, 32)
               if k <= batch and batch % k == 0 and (k == 1 or k*n*m <= MAX_PARALLEL_PAIRS))


@jax.jit
def batch_keys(key):
    next_key, sample_key = jax.random.split(key)
    return next_key, jax.random.split(sample_key, BATCH_SIZE)


@functools.lru_cache(None)
def group_fn(n, m, method):
    f = base(n, m)
    @jax.jit
    def one(params, key):
        t, _ = f['teacher'](params, OBS, key, QARG)
        if method == 'baseline':
            value, g = jax.value_and_grad(f['loss'])(params, t)
            d = assignment(params, t)
            retained = jnp.array(1.)
        else:
            g, d, coeff = gradients(params, t, method)
            value = f['loss'](params, t)
            retained = coeff.sum()
        # Group scalars: gradient norm is computed AFTER batch aggregation.
        metric = jnp.concatenate((jnp.array([value, d['confidence'].mean(), retained,
            1/jnp.square(t['w']).sum(), t['w'].max()]),
            jnp.mean(jax.nn.one_hot(d['mode'], 3), axis=0)))
        return g, metric
    return one


@functools.lru_cache(None)
def aggregate_fn(n, m, method, micro=None):
    micro = micro or microbatch_size(n, m)
    assert BATCH_SIZE % micro == 0
    one = group_fn(n, m, method)
    @jax.jit
    def aggregate(params, keys):
        assert keys.shape[0] == BATCH_SIZE
        zero = jax.tree_util.tree_map(jnp.zeros_like, params)
        def body(carry, group_keys):
            acc, met = carry
            gs, ms = jax.vmap(one, in_axes=(None, 0))(params, group_keys)
            acc = jax.tree_util.tree_map(lambda a, x: a+x.sum(0)/BATCH_SIZE, acc, gs)
            return (acc, met+ms.sum(0)/BATCH_SIZE), None
        (g, metric), _ = jax.lax.scan(body, (zero, jnp.zeros(8)),
                                    keys.reshape(BATCH_SIZE//micro, micro, 2))
        return g, metric
    return aggregate


@functools.lru_cache(None)
def engine(n, m, method):
    assert method in METHODS
    aggregate = aggregate_fn(n, m, method)
    @jax.jit
    def step(state, key):
        next_key, keys = batch_keys(key)
        g, metric = aggregate(state.params, keys)
        # Never update params/Adam inside the group or microbatch loop.
        new = state.apply_gradients(grads=g)
        gn = jnp.sqrt(sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(g)))
        return new, next_key, jnp.concatenate((metric[:1], gn[None], metric[1:]))
    @jax.jit
    def block(state, key, count):
        def body(_, carry):
            s, k, total, _ = carry
            ns, nk, metric = step(s, k)
            return ns, nk, total+metric, metric
        ns, nk, total, last = jax.lax.fori_loop(0, count, body,
            (state, key, jnp.zeros(9), jnp.zeros(9)))
        return ns, nk, total/count, last
    return dict(step=step, block=block)
