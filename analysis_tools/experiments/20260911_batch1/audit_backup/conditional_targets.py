"""Fixed actor/latents, repeated proposal clouds: isolate random regression targets."""
import argparse,json,time
from pathlib import Path
import jax,jax.numpy as jnp,jax.scipy as jsp
import numpy as np
from flax import serialization
from analysis_batch1.core import config_for,initialize
from analysis_boltzmann.problems import make_problem
from optiq_dime.transport import TruncatedGaussianKDE,sinkhorn


def inspect(campaign,case,repetitions):
    cfg=config_for(case,'ver2',0);p=make_problem(case);template,_,_=initialize(cfg)
    state=serialization.from_bytes(template,(campaign/'runs'/(case+'_ver2_seed0')/'actor_20000.msgpack').read_bytes())
    n=2048
    latent=jax.random.normal(jax.random.PRNGKey(909001),(n,p.dim))
    raw=state.apply_fn({'params':state.params},jnp.zeros((n,1)),latent)[None]
    actor=jnp.clip(raw,-1,1)
    kde=TruncatedGaussianKDE.from_centers(actor,1.,.5)
    @jax.jit
    def draw_targets(key):
        candidates=kde.sample_stratified(key,1,False).reshape(1,n,p.dim)
        q=p.q(candidates,jnp,jsp.special.logsumexp)
        weights=jax.nn.softmax(q-kde.log_prob(candidates),axis=-1)
        cost=((actor[:,:,None,:]-candidates[:,None,:,:])**2).sum(-1)
        cost=cost/(cost.mean((-2,-1),keepdims=True)+1e-8)
        plan=sinkhorn(cost,weights,.05,30)
        normalized=plan/jnp.maximum(plan.sum(-1,keepdims=True),1e-20)
        return candidates[0,jnp.argmax(normalized[0],axis=-1)]
    targets=[];start=time.monotonic()
    for rep in range(repetitions):
        targets.append(np.asarray(draw_targets(jax.random.PRNGKey(909100+rep))).astype(np.float64))
    targets=np.stack(targets)
    x=np.asarray(raw[0]).astype(np.float64);mean_target=targets.mean(0)
    loss=float(np.mean(np.sum((targets-x[None])**2,axis=-1)))
    mean_residual=float(np.mean(np.sum((mean_target-x)**2,axis=-1)))
    target_variance=float(np.mean(np.sum((targets-mean_target[None])**2,axis=-1)))
    np.testing.assert_allclose(loss,mean_residual+target_variance,atol=1e-10)
    return dict(case=case,version='ver2',seed=0,checkpoint=20000,
        fixed_latent_seed=909001,proposal_repetitions=repetitions,
        actor_mean_q=float(p.q(np.asarray(actor[0])).mean()),random_target_mean_q=float(p.q(targets).mean()),
        mean_target_mean_q=float(p.q(mean_target).mean()),
        mse_to_random_targets=loss,mse_to_mean_target=mean_residual,
        conditional_target_variance=target_variance,
        target_variance_fraction=target_variance/loss,
        actor_target_mean_coordinate_rmse=float(np.sqrt(mean_residual/p.dim)),
        seconds=time.monotonic()-start,
        interpretation='All source latents held fixed; only proposals resampled. Finite-repeat conditional diagnostic, not a training intervention.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--repetitions',type=int,default=64)
    args=parser.parse_args();out=Path(args.out);rows=[]
    for case in ('separable_8','separable_4','modes2d_8'):
        row=inspect(Path(args.campaign),case,args.repetitions);rows.append(row)
        (out/'conditional_targets.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True);jax.clear_caches()
    (out/'CONDITIONAL_COMPLETE').write_text('No actor updates performed\n')
