"""1M real actions, 512 bins, inherited mode metrics and 1D Wasserstein."""
import numpy as np
from ..kl_diverse_targets_1d.evaluate import evaluate as gm_eval
from ..kl_nongmm_targets_1d.evaluate import evaluate as ng_eval
from ..kl_diverse_targets_1d.target import reference as gm_ref
from ..kl_nongmm_targets_1d.target import reference as ng_ref

class SampleView:
    def __init__(self,exp):self.exp=exp;self.cfg=exp.cfg
    def samples(self,count):
        if count<=32768:return self.exp.samples(count)
        chunk=self.cfg['final_sample_chunk'];parts=[]
        for i,start in enumerate(range(0,count,chunk)):
            parts.append(self.exp.samples(min(chunk,count-start),seed=197+104729*i))
        return tuple(np.concatenate([p[j] for p in parts],axis=0) for j in range(3))

def evaluate(exp,folder,step):
    ng=exp.cfg['target_kind']=='nongmm';fn=ng_eval if ng else gm_eval;ref=ng_ref if ng else gm_ref
    m=fn(SampleView(exp),folder,step)
    z=np.load(folder/f'samples_{step:06d}.npz');a=np.sort(z['actions'].ravel());values=[]
    for n in [65537,131073]:
        x=np.linspace(-10,10,n);emp=np.searchsorted(a,x,side='right')/len(a);target=ref(exp.cfg,x)[1]
        values.append(float(np.trapz(np.abs(emp-target),x)))
    m.update(wasserstein_1=values[-1],wasserstein_grid_difference=abs(values[-1]-values[0]),
             wasserstein_grid_points=131073,sampling='independent chunks:seed=197+104729*i' if len(a)>32768 else 'seed197')
    assert m['wasserstein_grid_difference']<2e-4,m
    return m
