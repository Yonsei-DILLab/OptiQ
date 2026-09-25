"""Policy-averaged Q spatial slice with fixed initial pose/velocity."""
import numpy as np


def diagnose(learner,output):
    import jax.numpy as jnp
    from .envs import make_one
    from .learners import evaluation_rng
    env=make_one('v1',0,reward_profile=learner.reward_profile,random_init=False)
    initial=env.reset();env.close()
    x=np.linspace(-9.8,1.8,59);y=np.linspace(-5.8,5.8,59)
    xx,yy=np.meshgrid(x,y)
    valid=~((xx>=-6)&(xx<=-2)&(yy>=-2)&(yy<=2))
    locations=np.stack([xx[valid],yy[valid]],axis=1)
    means=[];stds=[];twin_diffs=[]
    p=learner.model.policy
    with evaluation_rng(learner,96000):
        for start in range(0,len(locations),32):
            points=locations[start:start+32]
            obs=np.repeat(initial[None],len(points),axis=0);obs[:,:2]=points
            repeated=np.repeat(obs,64,axis=0)
            actions=learner.act(repeated,'native')
            qs=np.asarray(p.qf_state.apply_fn({'params':p.qf_state.params,'batch_stats':p.qf_state.batch_stats},
                jnp.asarray(repeated),jnp.asarray(actions),train=False))[...,0].T.reshape(len(points),64,2)
            q=qs.mean(axis=2)
            means.extend(q.mean(axis=1));stds.extend(q.std(axis=1));twin_diffs.extend(np.abs(qs[:,:,0]-qs[:,:,1]).mean(axis=1))
    mean=np.full(xx.shape,np.nan);std=mean.copy();disagreement=mean.copy()
    mean[valid]=means;std[valid]=stds;disagreement[valid]=twin_diffs
    np.savez_compressed(output/'spatial_q.npz',x=x,y=y,mean=mean,std=std,disagreement=disagreement,initial_obs=initial)
    upper=yy>0
    delta=mean-np.flipud(mean)
    return dict(grid_shape=list(mean.shape),valid_positions=int(valid.sum()),z_samples_per_state=64,
                q_aggregation='mean of twin critics, then mean over random-z mu-only actions',
                fixed='initial pose/joints/velocity; only observation x,y changed',
                conditional_noise=False,upper_minus_mirror_mean=float(np.nanmean(delta[upper])),
                caveat='Synthetic state slice, not rollout return; body-center wall mask only; states may be out of replay support')
