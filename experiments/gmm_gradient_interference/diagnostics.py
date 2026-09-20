"""Discarded one-update branches; training RNG/actor/Adam are read-only inputs."""
from .core import *
import time


def cosine(gram):
    norm=np.sqrt(np.maximum(np.diag(gram),0))
    den=norm[:,None]*norm[None,:]
    return np.divide(gram,den,out=np.zeros_like(gram),where=den>1e-30),norm


def probe(state,key,n,m,seed):
    start=time.monotonic();f=base(n,m)
    t,_=f['teacher'](state.params,OBS,key,QARG)  # exactly the NEXT training teacher
    gs=mode_grads(state.params,t)
    vectors=np.stack([np.asarray(ravel_pytree(jax.tree_util.tree_map(lambda x:x[i],gs))[0]) for i in range(3)])
    flat,unravel=ravel_pytree(state.params)
    full_grad=jax.tree_util.tree_map(lambda x:x.sum(0),gs)
    full=f['update'](state,t)[0]  # original loss/Adam, not a reassociated sum
    full_delta=ravel_pytree(full.params)[0]-flat
    length=jnp.linalg.norm(full_delta)
    zero=jax.tree_util.tree_map(jnp.zeros_like,state.params)
    branches=[('full_adam',full),('momentum_only',state.apply_gradients(grads=zero))]
    for i,label in enumerate(['left','center','right']):
        g=jax.tree_util.tree_map(lambda x:x[i],gs)
        branches.append((label+'_adam',state.apply_gradients(grads=g)))
    for i,label in enumerate(['left','center','right']):
        v=jnp.asarray(vectors[i]);delta=-length*v/jnp.maximum(jnp.linalg.norm(v),1e-30)
        branches.append((label+'_sgd_matched_norm',state.replace(params=unravel(flat+delta))))
    # Fixed latent identity persists through training; label based on pre-update purity.
    z=jax.random.normal(jax.random.PRNGKey(77000+seed),(2048,1))
    mu,ls=heads(state.params,z);prob=basin_prob(mu,ls)
    before_terms=np.asarray(mode_terms(state.params,t))
    rt=reference_teacher(z);before_ref=np.asarray(mode_terms(state.params,rt))
    arrays=dict(z=np.asarray(z),mu=np.asarray(mu),log_sigma=np.asarray(ls),basin_prob=np.asarray(prob),
                before_terms=before_terms,before_reference_terms=before_ref,teacher_b=np.asarray(t['b'][0,:,0]),
                teacher_u=np.asarray(t['u'][0,:,0]),teacher_w=np.asarray(t['w'][0]),
                training_z=np.asarray(t['z'][0]),teacher_Q=np.asarray(t['Q'][0]),teacher_log_q=np.asarray(t['log_q'][0]))
    gram=np.asarray(gram_tree(gs),np.float64);cos,norm=cosine(gram)
    arrays.update(gradient_gram=gram,gradient_cosine=cos,gradient_norm=norm)
    teacher_labels=np.digitize(arrays['teacher_b'],[-.3,.3])
    mode_count=np.bincount(teacher_labels,minlength=3)
    mode_mass=np.bincount(teacher_labels,weights=arrays['teacher_w'],minlength=3)
    arrays.update(teacher_mode_count=mode_count,teacher_mode_mass=mode_mass,
                  gradient_cosine_valid=(norm[:,None]*norm[None,:]>1e-30))
    # Head versus shared trunk blocks, all differentiated w.r.t. parameters.
    for label,subtree in [('mu_head',gs['mu']),('sigma_head',gs['log_std']),
                          ('trunk',{k:v for k,v in gs.items() if k not in ('mu','log_std')})]:
        g=np.asarray(gram_tree(subtree),np.float64);c,nn=cosine(g)
        arrays['cosine_'+label]=c;arrays['norm_'+label]=nn
        arrays['cosine_valid_'+label]=nn[:,None]*nn[None,:]>1e-30
    assignment,alpha,rowmode=responsibility(state.params,t)
    arrays.update(assignment_mode_mass=np.asarray(assignment),alpha=np.asarray(alpha),row_mode_fraction=np.asarray(rowmode))
    changes=[];reference_changes=[];prediction=[];delta_mu=[];delta_ls=[];after_probs=[];paramnorm=[]
    for name,new in branches:
        delta=np.asarray(ravel_pytree(new.params)[0]-flat)
        after=np.asarray(mode_terms(new.params,t));ref=np.asarray(mode_terms(new.params,rt))
        mm,ll=heads(new.params,z)
        changes.append(after-before_terms);reference_changes.append(ref-before_ref)
        prediction.append(vectors@delta);delta_mu.append(np.asarray(mm-mu));delta_ls.append(np.asarray(ll-ls))
        after_probs.append(np.asarray(basin_prob(mm,ll)));paramnorm.append(np.linalg.norm(delta))
    arrays.update(branch_names=np.array([x[0] for x in branches]),delta_mode_nll=np.asarray(changes),
        delta_reference_nll=np.asarray(reference_changes),first_order_delta_nll=np.asarray(prediction),
        delta_mu=np.asarray(delta_mu),delta_log_sigma=np.asarray(delta_ls),after_basin_prob=np.asarray(after_probs),
        parameter_step_norm=np.asarray(paramnorm))
    # Same-noise samples visualize baseline and each branch without density integration.
    samplekey=jax.random.PRNGKey(88000+seed)
    arrays['samples']=np.asarray(draw(state.params,samplekey))
    arrays['branch_histograms']=np.stack([np.histogram(np.asarray(draw(s.params,samplekey)),bins=np.linspace(-1,1,257))[0]/32768 for _,s in branches])
    pur=np.max(arrays['basin_prob'],axis=1);labels=np.where(pur>=.8,np.argmax(arrays['basin_prob'],1),3)
    arrays['fixed_latent_labels']=labels
    # Do not relabel groups after update: cross-mode movement concerns the OLD specialist group.
    move=np.full((len(branches),4),np.nan);mass_change=np.full_like(move,np.nan);counts=np.bincount(labels,minlength=4)
    for b in range(len(branches)):
        before_pos=np.tanh(arrays['mu']);after_pos=np.tanh(arrays['mu']+arrays['delta_mu'][b])
        for g in range(4):
            select=labels==g
            if select.any():
                move[b,g]=np.sqrt(np.mean((after_pos[select]-before_pos[select])**2))
                if g<3:mass_change[b,g]=np.mean(arrays['after_basin_prob'][b,select,g]-arrays['basin_prob'][select,g])
    arrays.update(specialist_counts=counts,cross_mode_position_rms=move,cross_mode_own_mass_change=mass_change)
    # Shared trunk and mean/sigma interference need not have the same sign.
    summary=dict(specialist_fraction=float((pur>=.8).mean()),specialist_counts=counts.tolist(),
        gradient_norm=norm.tolist(),gradient_cosine=cos.tolist(),
        teacher_mode_count=mode_count.tolist(),teacher_mode_mass=mode_mass.tolist(),
        gradient_mode_active=(norm>1e-15).tolist(),
        teacher_ess=float(1/np.square(arrays['teacher_w']).sum()),
        teacher_wmax=float(arrays['teacher_w'].max()),
        component_usage_ess=float(1/np.square(arrays['alpha']).sum()),
        underused_fraction=float(np.mean(arrays['alpha']<.1/n)),
        sigma_mean=float(np.exp(arrays['log_sigma']).mean()),
        full_adam_mode_delta=np.asarray(changes)[0].tolist(),
        full_adam_reference_delta=np.asarray(reference_changes)[0].tolist(),
        probe_seconds=time.monotonic()-start)
    return arrays,summary
