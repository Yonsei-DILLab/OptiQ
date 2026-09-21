"""True squared-Euclidean OT; never use sorted 1D transport in higher dimensions."""
import numpy as np
from .exact1d import monotone_plan

def exact_plan(x,y,w):
 if x.shape[1]==1:return monotone_plan(x[:,0],y[:,0],w)
 import ot
 x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);b=np.asarray(w,np.float64);b/=b.sum();a=np.full(len(x),1/len(x))
 cost=np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T,0)
 p,log=ot.emd(a,b,np.ascontiguousarray(cost),numItermax=1000000,log=True,numThreads=1,check_marginals=True)
 if log.get('warning'):raise RuntimeError('Exact OT solver did not converge: '+str(log['warning']))
 assert np.max(np.abs(p.sum(1)-a))<1e-7 and np.max(np.abs(p.sum(0)-b))<1e-7
 return p.astype(np.float32)
