"""Bounded additional IID prefixes for unresolved frozen actors; no new training."""
import argparse, hashlib, json, os, shutil, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .measure import make_accumulator, stable_minimum, rms, write, sha
from experiments.kl_direction_1d.core import Experiment


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    pcfg=json.loads((args.parent/'source/experiments/kl_direction_1d/config.json').read_text())
    cond=pcfg['conditions'][args.index%5];seed=pcfg['seeds'][args.index//5];name=f"{cond['method']}_L{cond['L']}_s{seed}"
    old=args.root/'runtime/refined_results'/name;out=args.root/'runtime/extended_results'/name
    assert (old/'COMPLETE.json').exists();out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    assert jax.default_backend()=='gpu'
    summary=json.loads((old/'SUMMARY.json').read_text());assert summary['reference_converged']
    d=dict(np.load(old/'scores.npz'));n=cfg['policy_probes'];actions=d['actions'];ref=d['reference']
    checkpoint=args.parent/'runtime/runs'/name/'checkpoint.msgpack';digest=sha(checkpoint)
    exp=Experiment(pcfg,cond['method'],cond['L'],seed);exp.restore(checkpoint)
    parameter_hash=hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()
    def bank_impl(params,z):
        mu,ls=jax.lax.map(lambda x:exp.components(params,x),z.reshape(-1,4096,1))
        return mu.reshape(-1,1),ls.reshape(-1,1)
    bank=jax.jit(bank_impl);acc=make_accumulator(cfg['kernel_chunk'])
    # Shared, independent diagnostic latent draws, never fed back into optimization.
    z=np.random.default_rng(761023).standard_normal((65536,1)).astype(np.float32)
    mu,ls=bank(exp.state.params,z);sigma=np.exp(np.asarray(ls).ravel())
    write(out/'SIGMA.json',dict(latent_seed=761023,latent_count=65536,minimum=float(sigma.min()),
        quantiles={str(p):float(np.quantile(sigma,p)) for p in [.1,.5,.9]},fraction_below_001=float(np.mean(sigma<.01)),
        fraction_at_floor=float(np.mean(sigma<=np.exp(-5)*(1+1e-5)))))
    start=time.time();oldLs=d['Ls'];levels=[2097152,4194304,8388608] if summary['strict_1pct_L'] is None else []
    prefix_error=0.
    if levels:
        assert oldLs[-1]==1048576
        extra=[];logs=[]
        for r in range(cfg['repeats']):
            z=np.random.default_rng(92813+r).standard_normal((levels[-1],1)).astype(np.float32)
            mu,ls=bank(exp.state.params,z);assert mu.dtype==jnp.float32
            weights=jnp.zeros(len(z),jnp.float64)
            if r==0:
                # Validate original prefix on the same actor before resuming sums.
                _,s0=acc(mu,ls,weights,actions,0,128//cfg['kernel_chunk'],(np.full(len(actions),-np.inf),np.zeros(len(actions))))
                prefix_error=float(np.max(np.abs(np.asarray(s0)-d['scores'][0,0])))
                assert prefix_error<1e-9,(name,prefix_error)
            carry=(d['log_density'][r,-1]+np.log(oldLs[-1]),d['scores'][r,-1]);prev=int(oldLs[-1]);ss=[];ll=[]
            for L in levels:
                carry=acc(mu,ls,weights,actions,prev//cfg['kernel_chunk'],L//cfg['kernel_chunk'],carry)
                ss.append(np.asarray(carry[1]));ll.append(np.asarray(carry[0])-np.log(L));prev=L
            extra.append(ss);logs.append(ll)
            write(out/'STATUS.json',dict(phase='MC_extension',repeat_done=r+1,repeats=cfg['repeats'],time=time.time()))
            print(json.dumps(dict(name=name,repeat=r+1,seconds=time.time()-start)),flush=True)
        d['scores']=np.concatenate((d['scores'],np.asarray(extra)),axis=1)
        d['log_density']=np.concatenate((d['log_density'],np.asarray(logs)),axis=1)
        d['Ls']=np.concatenate((oldLs,np.asarray(levels)))
    s=d['scores'];Ls=d['Ls'];scale=summary['reference_score_RMS'];gscale=float(rms(ref[n:]))
    errors=rms(s[:,:,:n]-ref[None,None,:n],axis=2)/(scale+1e-12)
    grid=rms(s[:,:,n:]-ref[None,None,n:],axis=2)/(gscale+1e-12)
    delta=rms(np.diff(s[:,:,:n],axis=1),axis=2)/(scale+1e-12)
    d.update(policy_errors=errors,grid_errors=grid,doubling_changes=delta)
    summary.update(strict_1pct_L=stable_minimum(Ls,errors,delta,.01),practical_5pct_L=stable_minimum(Ls,errors,delta,.05),
        grid_strict_1pct_L=stable_minimum(Ls,grid,rms(np.diff(s[:,:,n:],axis=1),axis=2)/(gscale+1e-12),.01),largest_L=int(Ls[-1]),
        precision_check_L=int(oldLs[-1]))
    summary['results']=[]
    for k,L in enumerate(Ls):
        summary['results'].append(dict(L=int(L),relative_RMSE_mean=float(errors[:,k].mean()),relative_RMSE_max=float(errors[:,k].max()),
            absolute_RMSE_mean=float(errors[:,k].mean()*scale),pointwise_abs_error_p99=float(np.quantile(np.abs(s[:,k,:n]-ref[:n]),.99)),
            empirical_bias_RMSE=float(rms(s[:,k,:n].mean(0)-ref[:n])),repeat_SD_RMS=float(rms(s[:,k,:n].std(0,ddof=1))),
            grid_relative_RMSE_max=float(grid[:,k].max()),next_doubling_relative_change_max=float(delta[:,k].max()) if k<len(Ls)-1 else None))
    np.savez_compressed(out/'scores.npz',**d);write(out/'SUMMARY.json',summary)
    for f in ['RUN.json','fixed_actions.npy','reference.npz','QUADRATURE.json','REFINEMENT.json']:shutil.copy2(old/f,out/f)
    manifest=json.loads((args.root/'EXTENSION_SOURCE_MANIFEST.json').read_text())
    receipt=dict(commit=manifest['commit'],job_id=os.environ.get('SLURM_JOB_ID'),hostname=os.uname().nodename,
        input_scores_sha256=sha(old/'scores.npz'),checkpoint_sha256=digest,extra_Ls=levels,seconds=time.time()-start,
        original_MC_prefix_unchanged=bool(np.array_equal(d['scores'][:,:len(oldLs)],np.load(old/'scores.npz')['scores'])),prefix_recompute_max_difference=prefix_error)
    assert sha(checkpoint)==digest
    assert hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()==parameter_hash
    write(out/'EXTENSION.json',receipt);write(out/'COMPLETE.json',dict(finished=time.time(),**receipt))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
