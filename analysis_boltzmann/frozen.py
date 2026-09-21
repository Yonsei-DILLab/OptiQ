"""Frozen-Q fit and repeated backup estimates with independent evaluation RNG."""
import argparse,csv,json,time
import numpy as np
import jax,jax.numpy as jnp,jax.scipy as jsp
from flax import serialization
from .problems import make_problem,local_estimates
from .shared import actor_state,dummy_critic,ot_update,begin
from optiq_dime.transport import sample_truncated_gaussian


def run(args):
    out=begin(args.out,args); p=make_problem(args.case)
    grid,mass,truth,resolution,delta=p.truth()
    np.savez_compressed(out/'reference.npz',grid=grid,mass=mass,truth=truth,resolution=resolution,convergence_delta=delta)
    state=actor_state(p.dim,args.seed)
    critic=dummy_critic(lambda obs,a:p.q(a,jnp,jsp.special.logsumexp))
    key=jax.random.PRNGKey(args.seed+1000)
    @jax.jit
    def sample(params,key,count_template):
        z=jax.random.normal(key,count_template.shape)
        return jnp.clip(state.apply_fn({'params':params},jnp.zeros((len(z),1)),z),-1,1)
    warm_start=time.monotonic()
    if args.initialization=='coverage':
        # Reference-informed warm start is explicitly NOT a default OptiQ result.
        # Deterministic inverse-CDF map preserves a latent-to-action association.
        rng=np.random.default_rng(args.seed+2000)
        z=rng.normal(size=(8192,p.dim)).astype('float32')
        from scipy.special import ndtr
        u=ndtr(z)
        if p.dim==1 or p.separable:
            target=np.interp(u,np.cumsum(mass),grid[:,0]).astype('float32')
        else:
            n=int(round(np.sqrt(len(grid)))); joint=mass.reshape(n,n)
            marginal=joint.sum(1); ix=np.searchsorted(np.cumsum(marginal),u[:,0]).clip(0,n-1)
            conditional=np.cumsum(joint[ix]/marginal[ix,None],axis=1)
            iy=(conditional<u[:,1,None]).sum(1).clip(0,n-1)
            target=grid[ix*n+iy].astype('float32')
        @jax.jit
        def warm(st,z,t):
            def loss(params): return jnp.mean((st.apply_fn({'params':params},jnp.zeros((len(z),1)),z)-t)**2)
            return st.apply_gradients(grads=jax.grad(loss)(st.params))
        for i in range(2000):
            ids=rng.integers(0,len(z),256); state=warm(state,jnp.asarray(z[ids]),jnp.asarray(target[ids]))
    (out/'warmup.json').write_text(json.dumps({'seconds':time.monotonic()-warm_start,'updates':2000 if args.initialization=='coverage' else 0}))
    rows=[]; train=[]; start=time.monotonic()
    def diagnose(step):
        rng=np.random.default_rng(args.seed+100000+step)
        evalkey=jax.random.PRNGKey(args.seed+200000+step)
        samples=np.asarray(sample(state.params,evalkey,jnp.zeros((32768,p.dim))))
        refsample=p.reference_sample(rng,(32768,),grid,mass)
        labels=p.labels(samples); reference_labels=p.labels(refsample)
        nlabels=max(labels.max(),reference_labels.max())+1
        observed=np.bincount(labels,minlength=nlabels)/len(labels)
        target_mass=np.bincount(reference_labels,minlength=nlabels)/len(reference_labels)
        np.savez_compressed(out/f'distribution_{step}.npz',samples=samples,mode_mass=observed,reference_mode_mass=target_mass)
        (out/f'actor_{step}.msgpack').write_bytes(serialization.to_bytes(state))
        for k in args.ks:
            estimates={}
            for name in ('optiq_raw','optiq_td','reference'):
                chunks=[]
                for offset in range(0,args.repetitions,32):
                    n=min(32,args.repetitions-offset)
                    evalkey,ak,nk=jax.random.split(evalkey,3)
                    if name=='reference': a=p.reference_sample(rng,(n*k,),grid,mass)
                    else:
                        a=sample(state.params,ak,jnp.zeros((n*k,p.dim)))
                        if name=='optiq_td': a=sample_truncated_gaussian(nk,a,repeats=1,std=.2,perturb_clip=.5)[:,0]
                        a=np.asarray(a)
                    chunks.append(p.q(a).reshape(n,k).mean(1))
                estimates[name]=np.concatenate(chunks)
            # Frozen local estimators need evaluation only once; same Q at all steps.
            if step==0:
                centers=[np.repeat(c[0],p.dim) for c in p.centers] if p.separable else p.centers
                for ci,center in enumerate(centers):
                    for mode in ('no_is','official_is','truncated_is'):
                        chunks=[]; mode_counts=np.zeros(nlabels)
                        for offset in range(0,args.repetitions,32):
                            v,candidates,weights=local_estimates(p,center,k,min(32,args.repetitions-offset),rng,mode)
                            chunks.append(v)
                            if k==50:
                                bins=p.labels(candidates).reshape(-1)
                                counts=np.bincount(bins,weights=weights.reshape(-1),minlength=nlabels)
                                if len(counts)>len(mode_counts): mode_counts=np.pad(mode_counts,(0,len(counts)-len(mode_counts)))
                                mode_counts[:len(counts)]+=counts
                        if k==50:
                            np.savez_compressed(out/f'local_modes_{mode}_center{ci}.npz',mode_mass=mode_counts/args.repetitions,reference_mode_mass=target_mass,center=center)
                        estimates[f'local_{mode}_center{ci}']=np.concatenate(chunks)
            np.savez_compressed(out/f'estimates_{step}_k{k}.npz',**estimates)
            for name,v in estimates.items():
                if not np.isfinite(v).all(): raise FloatingPointError(name)
                rows.append(dict(step=step,k=k,method=name,mean=v.mean(),bias=v.mean()-truth,std=v.std(ddof=1),rmse=np.sqrt(np.mean((v-truth)**2)),truth=truth,repetitions=len(v),seed=args.seed,case=args.case,initialization=args.initialization))
        with open(out/'estimates.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        print(json.dumps({'step':step,'elapsed':time.monotonic()-start,'case':args.case}),flush=True)
    checkpoints={0,100,1000,5000,args.updates}
    diagnose(0)
    obs=jnp.zeros((256,1))
    for step in range(1,args.updates+1):
        state,loss,key,metrics=ot_update(state,critic,obs,key)
        if step%100==0 or step==args.updates:
            value=float(loss)
            if not np.isfinite(value): raise FloatingPointError('actor loss')
            train.append({'step':step,'loss':value,'elapsed':time.monotonic()-start,'ess':float(metrics['source_ess_absolute'])})
        if step in checkpoints: diagnose(step)
    (out/'training.json').write_text(json.dumps(train,indent=2))
    (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--case',default='asymmetric_1.3_0.08'); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--initialization',choices=['default','coverage'],default='default'); ap.add_argument('--updates',type=int,default=20000)
    ap.add_argument('--repetitions',type=int,default=2000); ap.add_argument('--ks',type=int,nargs='+',default=[1,8,16,50,64,256,1024]); ap.add_argument('--out',required=True)
    run(ap.parse_args())
