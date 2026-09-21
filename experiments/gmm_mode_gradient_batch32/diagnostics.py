"""Representative one-group geometry plus actual 32-group update diagnostics."""
from .core import *
from experiments.gmm_mode_gradient.diagnostics import probe as single_probe
from experiments.gmm_gradient_interference.diagnostics import cosine


@functools.lru_cache(None)
def batch_mode_grads_fn(n, m):
    f = base(n, m)
    @jax.jit
    def measure(params, keys):
        zero = jax.tree_util.tree_map(lambda x: jnp.zeros((3,)+x.shape, x.dtype), params)
        def body(acc, key):
            t, _ = f['teacher'](params, OBS, key, QARG)
            g = mode_grads(params, t)
            d = assignment(params, t)
            acc = jax.tree_util.tree_map(lambda a, x: a+x/BATCH_SIZE, acc, g)
            counts = jax.nn.one_hot(d['labels'], 3)
            summary = (d['H'], d['alpha'], counts.sum(0), t['w'][0]@counts)
            return acc, summary
        return jax.lax.scan(body, zero, keys)
    return measure


def probe(state, key, n, m, seed, method):
    _, keys = batch_keys(key)
    single, summary = single_probe(state, keys[0], n, m, seed, method)
    arrays = {'single_group_'+k: v for k, v in single.items()}
    # H figure intentionally shows the first group, not a merger of 32 mixtures.
    for k in ('H', 'training_z', 'samples', 'z', 'mu', 'log_sigma', 'basin_prob'):
        arrays[k] = single[k]
    gs, stats = batch_mode_grads_fn(n, m)(state.params, keys)
    gram = np.asarray(gram_tree(gs), np.float64)
    cos, norm = cosine(gram)
    arrays.update(gradient_gram=gram, gradient_cosine=cos, gradient_norm=norm,
                  batch_H=np.asarray(stats[0]), batch_alpha=np.asarray(stats[1]),
                  batch_teacher_mode_count=np.asarray(stats[2]), batch_teacher_mode_mass=np.asarray(stats[3]),
                  diagnostic_batch_size=np.array(BATCH_SIZE), representative_group_index=np.array(0))
    z=jnp.asarray(single['z']); mu,ls=heads(state.params,z)
    rt=reference_teacher(z); before_ref=mode_terms(state.params,rt)
    moves=[]; norms=[]; hist=[]; refs=[]; metrics=[]
    for option in METHODS:
        new, _, metric=engine(n,m,option)['step'](state,key)
        mm,ll=heads(new.params,z)
        moves.append(np.stack([np.asarray(mm-mu),np.asarray(ll-ls)],axis=-1))
        norms.append(float(metric[1]));metrics.append(np.asarray(metric))
        s=np.asarray(draw(new.params,jax.random.PRNGKey(88000+seed)))
        hist.append(np.histogram(s,np.linspace(-1,1,257))[0]/len(s))
        refs.append(np.asarray(mode_terms(new.params,rt)-before_ref))
    arrays.update(routed_method_names=np.array(METHODS),routed_fixed_latent_delta=np.asarray(moves),
        routed_gradient_norm=np.asarray(norms),routed_branch_histograms=np.asarray(hist),
        routed_reference_mode_nll_delta=np.asarray(refs),routed_batch_metrics=np.asarray(metrics))
    output=dict(training_method=method,batch_size=BATCH_SIZE,
        diagnostic_scope='H represents group 0; gradient_gram and routed branches average all 32 groups',
        single_group_summary=summary,gradient_norm=norm.tolist(),gradient_cosine=cos.tolist(),
        teacher_mode_mass_mean=np.asarray(stats[3]).mean(0).tolist(),
        routed_gradient_norm=norms, microbatch_size=microbatch_size(n,m))
    return arrays,output
