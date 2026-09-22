import numpy as np
from scipy.optimize import linear_sum_assignment

def monge_indices(x,b,w,offset,order):
    """Exact 2D linear assignment to N systematic teacher representatives.

    This is Monge for the equal-weight empirical approximation, not the arbitrary
    original weighted M-atom measure. Randomize candidate order before systematic
    resampling so there is no artificial coordinate/mode order preference.
    """
    x,b,w=np.asarray(x,np.float64),np.asarray(b,np.float64),np.asarray(w,np.float64)
    order=np.asarray(order,np.int64);cdf=np.cumsum(w[order]/w.sum());cdf[-1]=1.
    selected=order[np.minimum(np.searchsorted(cdf,(np.arange(len(x))+float(offset))/len(x),side='right'),len(b)-1)]
    cost=((x[:,None,:]-b[selected][None,:,:])**2).sum(-1)
    rows,cols=linear_sum_assignment(cost)
    result=np.empty(len(x),np.int32);result[rows]=selected[cols]
    return result
