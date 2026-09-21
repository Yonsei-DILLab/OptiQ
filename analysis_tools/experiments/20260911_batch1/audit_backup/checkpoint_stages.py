"""Read-only CPU audit of trained actors: candidate weighting, OT, hard targets."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,jax.scipy as jsp
from flax import serialization
from analysis_batch1.core import config_for,initialize,update
from analysis_boltzmann.problems import make_problem
from optiq_dime.transport import TruncatedGaussianKDE,sinkhorn


def inspect(case,version,campaign,repetitions):
    cfg=config_for(case,version,0);p=make_problem(case)
    template,oracle,_=initialize(cfg)
    folder=campaign/'runs'/(case+'_'+version+'_seed0')
    actor=serialization.from_bytes(template,(folder/'actor_20000.msgpack').read_bytes())
    n=cfg['num_policy_samples'];r=cfg['proposals_per_policy_sample'];m=n*r
    @jax.jit
    def forward(params,key):
        _,lk,pk,_=jax.random.split(key,4)
        z=jax.random.normal(lk,(1,n,p.dim));obs=jnp.zeros((n,1))
        raw=actor.apply_fn({'params':params},obs,z.reshape(n,p.dim))[None]
        a=jnp.clip(raw,-1,1)
        kde=TruncatedGaussianKDE.from_centers(a,1.,.5)
        candidates=kde.sample_stratified(pk,r,cfg['include_anchor']).reshape(1,m,p.dim)
        q=p.q(candidates,jnp,jsp.special.logsumexp)
        weights=jax.nn.softmax(q-kde.log_prob(candidates),axis=-1)
        cost=((a[:,:,None,:]-candidates[:,None,:,:])**2).sum(-1)
        cost=cost/(cost.mean((-2,-1),keepdims=True)+1e-8)
        result={'actor_q':p.q(a,jnp,jsp.special.logsumexp).mean(),
                'candidate_uniform_q':q.mean(),'weighted_q':(weights*q).sum(),
                'ess':1/(weights**2).sum()}
        for iterations in (30,300):
            coupling=sinkhorn(cost,weights,.05,iterations)
            normalized=coupling/jnp.maximum(coupling.sum(-1,keepdims=True),1e-20)
            row_average=normalized.mean(axis=1)
            ids=jnp.argmax(normalized[0],axis=-1);targets=candidates[0,ids]
            prefix=str(iterations)+'_'
            result[prefix+'coupling_q']=(coupling.sum(1)*q).sum()
            result[prefix+'row_expected_q']=(row_average*q).sum()
            result[prefix+'hard_target_q']=p.q(targets,jnp,jsp.special.logsumexp).mean()
            result[prefix+'row_l1']=abs(coupling.sum(-1)-1/n).sum()
            result[prefix+'column_l1']=abs(coupling.sum(1)-weights).sum()
            result[prefix+'row_weights_tv']=abs(row_average-weights).sum()/2
            result[prefix+'loss']=((raw[0]-targets)**2).sum(-1).mean()
            result[prefix+'target_coordinate_valley_mass']=(abs(targets)<.2).mean()
        return result
    values=[];parity=None;start=time.monotonic()
    for rep in range(repetitions):
        key=jax.random.PRNGKey(7000000+rep)
        result={k:float(v) for k,v in forward(actor.params,key).items()}
        if rep==0:
            _,loss,_,metrics=update(actor,oracle,key,cfg,20000)
            # The update result is discarded; no actor, optimizer or training file is changed.
            parity=dict(reconstructed_loss=result['30_loss'],actual_loss=float(loss),
                        reconstructed_ess=result['ess'],actual_ess=float(metrics['source_ess_absolute']))
            np.testing.assert_allclose(result['30_loss'],float(loss),rtol=3e-4,atol=3e-5)
            np.testing.assert_allclose(result['ess'],float(metrics['source_ess_absolute']),rtol=3e-4,atol=3e-5)
        values.append(result)
    truth=float(np.load(campaign/'references'/(case+'.npz'))['truth'])
    result=dict(case=case,version=version,seed=0,checkpoint=20000,repetitions=repetitions,reference=truth,
        mean={k:float(np.mean([x[k] for x in values])) for k in values[0]},
        std={k:float(np.std([x[k] for x in values],ddof=1)) for k in values[0]},
        parity=parity,seconds=time.monotonic()-start)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--repetitions',type=int,default=8)
    args=parser.parse_args();campaign=Path(args.campaign);out=Path(args.out);out.mkdir(exist_ok=True)
    rows=[]
    for case in ('separable_8','separable_4','modes2d_8'):
        for version in ('ver1','ver2'):
            rows.append(inspect(case,version,campaign,args.repetitions))
            (out/'stages.json').write_text(json.dumps(rows,indent=2))
            jax.clear_caches()
    (out/'COMPLETE').write_text('Read-only checkpoint audit; no retraining or retained updates\n')
