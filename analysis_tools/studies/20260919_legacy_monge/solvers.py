"""Exact 1D empirical Monge assignment after explicit target quantization."""
import jax
import jax.numpy as jnp

def sorted_data(x,y,w):
    assert x.shape[1]==y.shape[1]==1, 'This pilot implements 1D squared-cost Monge only'
    ri=jnp.argsort(x[:,0],stable=True);ci=jnp.argsort(y[:,0],stable=True)
    mass=w[ci]/w.sum();cdf=jnp.cumsum(mass).at[-1].set(1.)
    return ri,ci,cdf

def monge_plan(x,y,w):
    """One match per indexed latent particle to N equal-mass target representatives.

    Representatives are original candidate values at midpoint target quantiles.
    Sorted matching is the globally minimum squared-cost bijection in 1D.
    The original arbitrary w is approximated, NOT enforced exactly.
    """
    n=len(x);m=len(y);ri,ci,cdf=sorted_data(x,y,w)
    selected=jnp.minimum(jnp.searchsorted(cdf,(jnp.arange(n)+.5)/n,side='left'),m-1)
    return jnp.zeros((n,m),w.dtype).at[ri,ci[selected]].set(1./n)

def exact_1d_plan(x,y,w):
    """Kantorovich plan to the original weighted candidate measure, allowing splits."""
    n=len(x);m=len(y);ri,ci,cdf=sorted_data(x,y,w)
    lo=jnp.concatenate((jnp.zeros(1,w.dtype),cdf[:-1]));hi=(jnp.arange(n)+1)/n
    sorted_plan=jnp.maximum(jnp.minimum(hi[:,None],cdf[None])-jnp.maximum((hi-1/n)[:,None],lo[None]),0.)
    return jnp.zeros((n,m),w.dtype).at[ri[:,None],ci[None,:]].set(sorted_plan)

def hard_plan(p):
    return jax.nn.one_hot(jnp.argmax(p,axis=-1),p.shape[-1],dtype=p.dtype)/p.shape[-2]

def assignment_metrics(x,b,w,p):
    n=len(x);a=hard_plan(p);selected_w=a.sum(0);order=jnp.argsort(b[:,0],stable=True)
    cd=jnp.cumsum(selected_w[order]-w[order]);dx=jnp.diff(b[order,0])
    return dict(selected_teacher_cdf_error=jnp.max(jnp.abs(cd)),
      selected_teacher_w1=jnp.sum(jnp.abs(cd[:-1])*dx),
      selected_teacher_atom_tv=.5*jnp.abs(selected_w-w).sum(),
      plan_column_l1=jnp.abs(p.sum(0)-w).sum(),plan_row_l1=jnp.abs(p.sum(1)-1/n).sum(),
      selected_cost=(a*jnp.square(x[:,None]-b[None]).sum(-1)).sum(),
      selected_unique=(selected_w>0).sum(),
      source_tie_fraction=jnp.mean(jnp.diff(jnp.sort(x[:,0]))==0))
