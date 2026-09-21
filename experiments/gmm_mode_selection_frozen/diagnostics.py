"""Actual batch gradient diagnostics; group-0 heatmaps are explicitly distinguished."""
from .core import *
from experiments.gmm_gradient_interference.diagnostics import cosine

@functools.lru_cache(None)
def batch_mode_grads_fn(name,n,m):
    f=functions(name,n,m);k=f['k']
    @jax.jit
    def measure(params,keys):
        zero=jax.tree_util.tree_map(lambda x:jnp.zeros((k,)+x.shape,x.dtype),params)
        def body(acc,key):
            t,_=f['teacher'](params,key);g=jax.jacrev(f['terms'])(params,t);d=f['assignment'](params,t)
            acc=jax.tree_util.tree_map(lambda a,x:a+x/BATCH_SIZE,acc,g)
            labels=jax.nn.one_hot(d['labels'],k)
            return acc,(d['H'],d['alpha'],labels.sum(0),t['w'][0]@labels)
        return jax.lax.scan(body,zero,keys)
    return measure

def probe(state,key,name,n,m,seed,method):
    f=functions(name,n,m);k=f['k'];_,keys=batch_keys(key)
    t,_=f['teacher'](state.params,keys[0]);d=f['assignment'](state.params,t)
    tm,tl=heads(state.params,t['z'][0]);delta=tm[:,0,None]-t['u'][0,:,0][None,:];inv=jnp.exp(-2*tl[:,0,None])
    gout=jnp.stack((delta*inv,1-jnp.square(delta)*inv),axis=-1)
    gm=jnp.einsum('ij,ijc,jm->mic',d['joint'],gout,jax.nn.one_hot(d['labels'],k))
    z=jax.random.normal(jax.random.PRNGKey(77000+seed),(2048,1));mu,ls=heads(state.params,z)
    samples=np.asarray(draw(state.params,jax.random.PRNGKey(88000+seed)));edges=np.asarray(f['ref']['edges'])
    gs,stats=batch_mode_grads_fn(name,n,m)(state.params,keys)
    gram=np.asarray(sum(x.reshape(k,-1)@x.reshape(k,-1).T for x in jax.tree_util.tree_leaves(gs)),np.float64)
    cos,norm=cosine(gram)
    arrays=dict(H=np.asarray(d['H']),joint=np.asarray(d['joint']),alpha=np.asarray(d['alpha']),
        assigned_mode=np.asarray(d['mode']),confidence=np.asarray(d['confidence']),candidate_labels=np.asarray(d['labels']),
        training_z=np.asarray(t['z'][0]),training_mu=np.asarray(tm),training_log_sigma=np.asarray(tl),
        candidates=np.asarray(t['b'][0,:,0]),Q=np.asarray(t['Q'][0]),log_q=np.asarray(t['log_q'][0]),
        logits=np.asarray(t['logits'][0]),weights=np.asarray(t['w'][0]),output_mode_gradients=np.asarray(gm),
        z=np.asarray(z),mu=np.asarray(mu),log_sigma=np.asarray(ls),basin_prob=np.asarray(conditional_basin_prob(mu,ls,name)),
        samples=samples,edges=edges,target_bin_mass=np.asarray(f['ref']['bin_mass']),
        gradient_gram=gram,gradient_cosine=cos,gradient_norm=norm,
        batch_H=np.asarray(stats[0]),batch_alpha=np.asarray(stats[1]),batch_teacher_mode_count=np.asarray(stats[2]),
        batch_teacher_mode_mass=np.asarray(stats[3]),diagnostic_batch_size=np.array(BATCH_SIZE),representative_group_index=np.array(0))
    moves=[];hists=[];metrics=[]
    for option in METHODS:
        new,_,metric=engine(name,n,m,option)['step'](state,key)
        mm,ll=heads(new.params,z);moves.append(np.stack([np.asarray(mm-mu),np.asarray(ll-ls)],axis=-1));metrics.append(np.asarray(metric))
        s=np.asarray(draw(new.params,jax.random.PRNGKey(88000+seed)))
        hists.append(np.histogram(s,edges)[0]/len(s))
    arrays.update(routed_method_names=np.array(METHODS),routed_fixed_latent_delta=np.asarray(moves),
        routed_branch_histograms=np.asarray(hists),routed_batch_metrics=np.asarray(metrics))
    summary=dict(training_method=method,landscape=name,batch_size=BATCH_SIZE,microbatch_size=microbatch_size(n,m),
        diagnostic_scope='Group 0: joint/H/output gradients; batch 128: gradient Gram and paired Adam branches',
        gradient_norm=norm.tolist(),gradient_cosine=cos.tolist(),teacher_mode_mass_mean=np.asarray(stats[3]).mean(0).tolist())
    return arrays,summary
