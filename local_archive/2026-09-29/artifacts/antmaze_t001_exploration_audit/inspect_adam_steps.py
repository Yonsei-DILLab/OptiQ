"""Reconstruct saved Adam update magnitudes; no forward/backward or training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json
import sys
import numpy as np
import torch
import flax.serialization
torch.set_num_threads(1)

def leaves(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):yield from leaves(tree[key],prefix+'/'+key)
    else:yield prefix,np.asarray(tree,dtype=np.float64).ravel()

root=Path(sys.argv[1]);out=[]
for run in sorted(root.iterdir()):
    checkpoints=sorted((run/'policy-checkpoints').glob('*/policy.pt'))
    if not checkpoints:continue
    path=checkpoints[-1]
    cp=torch.load(path,map_location='cpu',weights_only=False)
    raw=flax.serialization.msgpack_restore(cp['learner']['policy'])
    opt=cp['config']['native']['alg']['optimizer']
    result=dict(run=run.name,step=cp['step'],source=cp['config']['source_commit'],
        method='Saved Adam first/second moments, constant LR, no gradient clip; float64 reconstruction of last nominal update before parameter rounding')
    for name in ['actor','critic']:
        state=raw[name];adam=state['opt_state']['0'];count=int(adam['count'])
        assert set(adam)=={'count','mu','nu'} and not state['opt_state']['1']
        assert opt['ac_grad_norm'] is None
        lr=opt['lr_'+name];b1=opt[name+'_b1'];b2=opt[name+'_b2'];eps=1e-8
        mu=dict(leaves(adam['mu']));nu=dict(leaves(adam['nu']));params=dict(leaves(state['params']))
        assert mu.keys()==nu.keys()==params.keys()
        gradients=[];updates=[];weights=[]
        for key in mu:
            m=mu[key]/(1-b1**count);v=nu[key]/(1-b2**count)
            scale=np.sqrt(v)
            update=-lr*m/(scale+eps)
            gradients.append(scale);updates.append(update);weights.append(params[key])
        g=np.concatenate(gradients);u=np.concatenate(updates);p=np.concatenate(weights)
        result[name]=dict(lr=lr,step=count,parameter_count=len(p),
            gradient_second_moment_rms=float(np.sqrt(np.mean(g*g))),
            adam_sqrt_v_median=float(np.median(g)),
            epsilon_dominant_fraction=float(np.mean(g<eps)),
            reconstructed_update_rms=float(np.sqrt(np.mean(u*u))),
            parameter_rms=float(np.sqrt(np.mean(p*p))),
            relative_update_l2=float(np.linalg.norm(u)/np.linalg.norm(p-u)),
            reconstructed_zero_update_fraction=float(np.mean(u==0)))
    out.append(result)
print('AUDIT_JSON='+json.dumps(out,allow_nan=False))
