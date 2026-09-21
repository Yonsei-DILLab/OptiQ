"""Frozen-teacher routing and discarded paired-Adam branch diagnostics."""
from .core import *
from experiments.gmm_gradient_interference.diagnostics import probe as original_probe


def probe(state,key,n,m,seed,method):
    arrays,summary=original_probe(state,key,n,m,seed)
    t,_=base(n,m)['teacher'](state.params,OBS,key,QARG)
    d=assignment(state.params,t);mu,ls=heads(state.params,t['z'][0]);u=t['u'][0,:,0]
    delta=mu[:,0,None]-u[None,:];inv=jnp.exp(-2*ls[:,0,None])
    # Three gradients wrt each output (mu_i, log sigma_i), before network VJP.
    output=jnp.stack((delta*inv,1-jnp.square(delta)*inv),axis=-1)
    gm=jnp.einsum('ij,ijc,jm->mic',d['joint'],output,jax.nn.one_hot(d['labels'],3))
    arrays.update(output_mode_gradients=np.asarray(gm),H=np.asarray(d['H']),
        assigned_mode=np.asarray(d['mode']),confidence=np.asarray(d['confidence']),
        candidate_labels=np.asarray(d['labels']))
    eval_z=jnp.asarray(arrays['z']);before_mu,before_ls=heads(state.params,eval_z)
    changes=[];before_loss=np.asarray(mode_terms(state.params,t));norms=[];hist=[]
    for option in METHODS:
        g,_,coeff=gradients(state.params,t,option)
        new=engine(n,m,option)['update'](state,t)[0]
        am,al=heads(new.params,eval_z)
        changes.append(np.stack([np.asarray(am-before_mu),np.asarray(al-before_ls)],axis=-1))
        norms.append(float(jnp.sqrt(sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(g)))))
        samples=np.asarray(draw(new.params,jax.random.PRNGKey(88000+seed)))
        hist.append(np.histogram(samples,np.linspace(-1,1,257))[0]/len(samples))
        arrays['routed_'+option+'_mode_nll_delta']=np.asarray(mode_terms(new.params,t))-before_loss
        arrays['routed_'+option+'_retained_mass']=np.asarray(coeff.sum())
    arrays.update(routed_method_names=np.array(METHODS),routed_fixed_latent_delta=np.asarray(changes),
        routed_gradient_norm=np.asarray(norms),routed_branch_histograms=np.asarray(hist))
    winner=np.asarray(d['mode']);confidence=np.asarray(d['confidence'])
    summary.update(training_method=method,assignment_counts=np.bincount(winner,minlength=3).tolist(),
        confidence_mean=float(confidence.mean()),confidence_min=float(confidence.min()),
        assignment_tie_fraction=float(np.mean(np.sum(np.asarray(d['H'])==np.asarray(d['H']).max(1,keepdims=True),axis=1)>1)),
        assignment_empty_modes=int(np.sum(np.bincount(winner,minlength=3)==0)))
    return arrays,summary
