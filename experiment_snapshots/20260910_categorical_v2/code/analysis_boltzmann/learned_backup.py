"""Common frozen MoveCar critic: production actor vs local estimators and quadrature.

Uses target twin-min Q for every estimator. The actor is not refit, so its
mean-Q / target-Q mismatch is intentionally retained and labelled.
"""
import argparse,csv,json
import numpy as np
import jax,jax.numpy as jnp
from scipy.special import logsumexp
from .movecar import make_agent
from .shared import begin
from .quadrature_check import reference,precise_q
from optiq_dime.transport import sample_truncated_gaussian


def run(args):
    out=begin(args.out,args); agent=make_agent('optiq',args.seed); agent.load(args.checkpoint)
    qstate=agent.model.policy.qf_state
    q=precise_q(qstate)
    (out/'reference_precision.json').write_text(json.dumps({'q_evaluation':'float64 evaluation of saved float32 weights, used by every comparator','actor_sampling':'unchanged float32','tolerance':1e-4,'consecutive_refinements':2},indent=2))
    rng=np.random.default_rng(args.seed+61000); key=jax.random.PRNGKey(args.seed+62000)
    rows=[]; reference_history=[]
    for state in np.linspace(0,10,args.states):
        truth,center,n,history=reference(q,state)
        reference_history.append({'state':float(state),'history':history})
        (out/'quadrature_history.json').write_text(json.dumps(reference_history,indent=2))
        for k in args.ks:
            for method in ('optiq_raw','optiq_td','local_no_is','local_official_is'):
                estimates=[]
                for offset in range(0,args.repetitions,32):
                    count=min(32,args.repetitions-offset); obs=np.full((count*k,1),state,np.float32)
                    key,ak,nk=jax.random.split(key,3)
                    if method.startswith('optiq'):
                        act=agent.act(obs,ak)
                        if method=='optiq_td': act=sample_truncated_gaussian(nk,act,1,.2,.5)[:,0]
                        values=np.asarray(q(obs,act)).reshape(count,k); estimate=values.mean(1)
                    else:
                        noise=rng.normal(0,.2,(count,k,1)); act=np.clip(center+np.clip(noise,-.5,.5),-1,1).reshape(-1,1).astype('float32')
                        values=np.asarray(q(obs,act)).reshape(count,k); logits=values/.25
                        if method=='local_official_is': logits-=(-.5*(noise[...,0]/.2)**2-np.log(.2*np.sqrt(2*np.pi)))
                        w=np.exp(logits-logsumexp(logits,axis=1,keepdims=True)); estimate=(w*values).sum(1)
                    estimates.extend(estimate.tolist())
                v=np.array(estimates)
                rows.append(dict(state=float(state),k=k,method=method,truth=truth,mean=v.mean(),bias=v.mean()-truth,rmse=np.sqrt(np.mean((v-truth)**2)),seed=args.seed,reference_n=n))
    with open(out/'learned_backup.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (out/'INTERPRETATION.txt').write_text('All methods evaluate the same target twin-min Q. Production actor was trained with live twin-mean Q. Local center is oracle global argmax on the reference grid.\n')
    (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--states',type=int,default=32); ap.add_argument('--repetitions',type=int,default=2000); ap.add_argument('--ks',nargs='+',type=int,default=[1,50,256]); ap.add_argument('--out',required=True); run(ap.parse_args())
