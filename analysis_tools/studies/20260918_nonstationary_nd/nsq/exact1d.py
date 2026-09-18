"""Exact 1D transport by O(N+M) probability-interval overlap in float64.

Compute sorted mass intervals on CPU without changing JAX actor precision.
Return a dense float32 plan for the existing GPU conditional NLL.
"""
import numpy as np

def monotone_plan(source,target,weights):
    source=np.asarray(source);target=np.asarray(target);w=np.asarray(weights,np.float64)
    ri=np.argsort(source,kind='stable');ci=np.argsort(target,kind='stable')
    w=w[ci]/w.sum();n=len(source);m=len(target)
    row_hi=np.arange(1,n+1,dtype=np.float64)/n
    col_hi=np.cumsum(w);col_hi[-1]=1.
    cuts=np.unique(np.concatenate(([0.],row_hi,col_hi)))
    left=cuts[:-1];mass=np.diff(cuts)
    rows=np.searchsorted(row_hi,left,side='right')
    cols=np.searchsorted(col_hi,left,side='right')
    assert np.all(rows<n) and np.all(cols<m)
    P=np.zeros((n,m),np.float32)
    P[ri[rows],ci[cols]]=mass.astype(np.float32)
    return P
