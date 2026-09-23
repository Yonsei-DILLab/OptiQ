"""Frozen-checkpoint score convergence, IID banks and latent quadrature."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss
from experiments.kl_direction_1d.core import Experiment,target_score

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def rms(x,axis=None):return np.sqrt(np.mean(np.square(x),axis=axis))

def make_accumulator(chunk=128,dtype=jnp.float64):
    def acc(mu,ls,logw,a,start,end,carry):
        mu=jnp.asarray(mu,dtype).reshape(-1,chunk,1)
        ls=jnp.asarray(ls,dtype).reshape(-1,chunk,1)
        logw=jnp.asarray(logw,dtype).reshape(-1,chunk,1)
        a=jnp.asarray(a,dtype).reshape(1,-1)
        carry=tuple(jnp.asarray(x,dtype) for x in carry)
        def body(b,c):
            old,score=c;m=mu[b];l=ls[b];inv=jnp.exp(-l)
            lognorm=jnp.log(jsp.special.ndtr((1-m)*inv)-jsp.special.ndtr((-1-m)*inv))
            ell=-.5*((a-m)*inv)**2-l-.5*jnp.log(2*jnp.pi)-lognorm+logw[b]
            block=jsp.special.logsumexp(ell,axis=0)
            ds=(jax.nn.softmax(ell,axis=0)*((m-a)*inv**2)).sum(0)
            total=jnp.logaddexp(old,block)
            score=jnp.exp(old-total)*score+jnp.exp(block-total)*ds
            return total,score
        return jax.lax.fori_loop(start,end,body,carry)
    return jax.jit(acc)

def quadrature_nodes(panels,order,limit):
    x,w=leggauss(order);edges=np.linspace(-limit,limit,panels+1)
    z=((edges[:-1,None]+edges[1:,None])/2+(edges[1:]-edges[:-1])[:,None]/2*x).ravel()
    weights=((edges[1:]-edges[:-1])[:,None]/2*w).ravel()
    return z.astype(np.float32)[:,None],np.log(weights)-.5*z*z-.5*np.log(2*np.pi)

def stable_minimum(Ls,errors,deltas,tolerance):
    # At least two successive L values; all observed larger banks must pass.
    for k in range(len(Ls)-1):
        if np.max(errors[:,k:])<=tolerance and np.max(deltas[:,k:])<=tolerance:return int(Ls[k])
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    pcfg=json.loads((args.parent/'source/experiments/kl_direction_1d/config.json').read_text())
    cond=pcfg['conditions'][args.index%5];seed=pcfg['seeds'][args.index//5]
    name=f"{cond['method']}_L{cond['L']}_s{seed}";src=args.parent/'runtime/runs'/name
    out=args.root/'runtime/results'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    assert (src/'COMPLETE.json').exists(),f'Parent training not complete: {name}'
    run=json.loads((src/'RUN.json').read_text());assert run['source_commit']==cfg['parent_source_commit']
    assert jax.default_backend()=='gpu','GPU allocation required'
    checkpoint=src/'checkpoint.msgpack';digest=sha(checkpoint)
    exp=Experiment(pcfg,cond['method'],cond['L'],seed);exp.restore(checkpoint)
    assert int(exp.state.step)==20000
    parameter_digest=hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()
    # Use saved native float32 samples; enabling x64 must NOT change sampling.
    saved=np.load(src/'samples_20000.npz');policy=saved['actions'].ravel()[:cfg['policy_probes']].astype(np.float32)
    grid=np.linspace(*cfg['grid_range'],cfg['grid_probes']).astype(np.float32)
    actions=np.concatenate((policy,grid));n=len(policy);count=len(actions)
    np.save(out/'fixed_actions.npy',actions)
    zeros=(jnp.full((count,),-jnp.inf,dtype=jnp.float64),jnp.zeros(count,dtype=jnp.float64))
    accumulator=make_accumulator(cfg['kernel_chunk'])
    def bank_impl(params,z):
        # Flax parameters and z remain float32, exactly the trained actor.
        mu,ls=jax.lax.map(lambda x:exp.components(params,x),z.reshape(-1,cfg['actor_chunk'],1))
        return mu.reshape(-1,1),ls.reshape(-1,1)
    bank_fn=jax.jit(bank_impl)
    def bank(z):
        size=len(z);pad=(-size)%cfg['actor_chunk']
        z=np.pad(np.asarray(z,np.float32),((0,pad),(0,0)))
        mu,ls=bank_fn(exp.state.params,jnp.asarray(z))
        assert mu.dtype==jnp.float32 and ls.dtype==jnp.float32
        return mu[:size],ls[:size]
    manifest=json.loads((args.root/'SOURCE_MANIFEST.json').read_text())
    metadata=dict(name=name,index=args.index,checkpoint_sha256=digest,checkpoint_source_commit=run['source_commit'],
        analysis_commit=manifest['commit'],config=cfg,parent_condition=cond,seed=seed,
        hostname=os.uname().nodename,job_id=os.environ.get('SLURM_JOB_ID'),
        device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,
        action_sha256=sha(out/'fixed_actions.npy'),started=time.time())
    write(out/'RUN.json',metadata)
    write(out/'STATUS.json',dict(phase='quadrature',time=time.time()))
    previous=None;stable=0;quad=[];reference_ok=False
    for panels in cfg['quadrature_panels']:
        z,lw=quadrature_nodes(panels,cfg['quadrature_order'],cfg['latent_limit']);mu,ls=bank(z)
        logp,s=accumulator(mu,ls,lw,actions,0,len(z)//cfg['kernel_chunk'],zeros)
        logp,s=np.asarray(logp),np.asarray(s)
        # Bound the omitted latent tail using support and minimum sigma.
        tail=2*ndtr(-cfg['latent_limit']);kmax=2/(np.exp(-5)*np.sqrt(2*np.pi));numerator_bound=tail*kmax*2*np.exp(10)
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            tail_bound=(numerator_bound+np.abs(s)*tail*kmax)/np.maximum(np.exp(logp)-tail*kmax,0)
        record=dict(nodes=len(z),max_tail_score_bound=float(np.max(tail_bound)))
        if previous is not None:
            diff=s-previous;criterion=np.abs(diff)/(cfg['reference_atol']+cfg['reference_rtol']*np.abs(s))
            good=np.all(criterion<=1) and np.all(tail_bound<1e-5)
            stable=stable+1 if good else 0
            record.update(max_score_change=float(np.max(np.abs(diff))),policy_relative_RMSE=float(rms(diff[:n])/(rms(s[:n])+1e-12)),passed_refinement=bool(good))
        quad.append(record);print(json.dumps(dict(name=name,phase='quadrature',**record)),flush=True)
        reference=s;reference_logp=logp;previous=s
        if stable>=2:reference_ok=True;break
    write(out/'QUADRATURE.json',dict(converged=reference_ok,levels=quad))
    np.savez_compressed(out/'reference.npz',actions=actions,score=reference,log_density=reference_logp,tail_score_error_bound=tail_bound)
    scale=float(rms(reference[:n]));gridscale=float(rms(reference[n:]))
    Ls=[L for L in cfg['Ls'] if L<=cfg['L_initial_max']]
    scores=[];logps=[];carry_list=[];timings=[]
    for r in range(cfg['repeats']):
        started=time.perf_counter();z=np.random.default_rng(92813+r).standard_normal((cfg['L_initial_max'],1)).astype(np.float32)
        mu,ls=bank(z);weights=jnp.zeros(len(z),dtype=jnp.float64);carry=zeros;ss=[];ll=[];prevL=0
        for L in Ls:
            carry=accumulator(mu,ls,weights,actions,prevL//cfg['kernel_chunk'],L//cfg['kernel_chunk'],carry)
            ll.append(np.asarray(carry[0])-np.log(L));ss.append(np.asarray(carry[1]));prevL=L
        scores.append(ss);logps.append(ll);carry_list.append(tuple(np.asarray(x) for x in carry));timings.append(time.perf_counter()-started)
        write(out/'STATUS.json',dict(phase='MC_initial',repeat_done=r+1,repeats=cfg['repeats'],Lmax=max(Ls),time=time.time()))
        print(json.dumps(dict(name=name,repeat=r,Lmax=max(Ls),seconds=timings[-1])),flush=True)
    scores=np.asarray(scores);logps=np.asarray(logps)
    errors=rms(scores[:,:,:n]-reference[None,None,:n],axis=2)/(scale+1e-12)
    deltas=rms(np.diff(scores[:,:,:n],axis=1),axis=2)/(scale+1e-12)
    strict=stable_minimum(Ls,errors,deltas,cfg['strict_nrmse']) if reference_ok else None
    if strict is None:
        extended=[L for L in cfg['Ls'] if L>max(Ls)];new_scores=[];new_logs=[]
        for r in range(cfg['repeats']):
            z=np.random.default_rng(92813+r).standard_normal((cfg['L_adaptive_max'],1)).astype(np.float32)
            mu,ls=bank(z);weights=jnp.zeros(len(z),dtype=jnp.float64);carry=carry_list[r];prevL=max(Ls);ss=[];ll=[]
            for L in extended:
                carry=accumulator(mu,ls,weights,actions,prevL//cfg['kernel_chunk'],L//cfg['kernel_chunk'],carry)
                ss.append(np.asarray(carry[1]));ll.append(np.asarray(carry[0])-np.log(L));prevL=L
            new_scores.append(ss);new_logs.append(ll)
            write(out/'STATUS.json',dict(phase='MC_extended',repeat_done=r+1,Lmax=max(extended),time=time.time()))
        scores=np.concatenate((scores,np.asarray(new_scores)),axis=1);logps=np.concatenate((logps,np.asarray(new_logs)),axis=1);Ls+=extended
    # Numerical precision check on the very same largest bank, repeat0.
    z=np.random.default_rng(92813).standard_normal((max(Ls),1)).astype(np.float32);mu,ls=bank(z)
    acc32=make_accumulator(cfg['kernel_chunk'],jnp.float32)
    _,score32=acc32(mu,ls,np.zeros(len(z),np.float32),actions,0,len(z)//cfg['kernel_chunk'],zeros)
    precision_error=float(rms(np.asarray(score32)[:n]-scores[0,-1,:n])/(scale+1e-12))
    errors=rms(scores[:,:,:n]-reference[None,None,:n],axis=2)/(scale+1e-12)
    grid_errors=rms(scores[:,:,n:]-reference[None,None,n:],axis=2)/(gridscale+1e-12)
    deltas=rms(np.diff(scores[:,:,:n],axis=1),axis=2)/(scale+1e-12)
    results=[]
    for k,L in enumerate(Ls):
        bias=scores[:,k,:n].mean(0)-reference[:n]
        results.append(dict(L=L,relative_RMSE_mean=float(errors[:,k].mean()),relative_RMSE_max=float(errors[:,k].max()),
            absolute_RMSE_mean=float(errors[:,k].mean()*scale),pointwise_abs_error_p99=float(np.quantile(np.abs(scores[:,k,:n]-reference[:n]),.99)),
            empirical_bias_RMSE=float(rms(bias)),repeat_SD_RMS=float(rms(scores[:,k,:n].std(0,ddof=1))),
            grid_relative_RMSE_max=float(grid_errors[:,k].max()),
            next_doubling_relative_change_max=float(deltas[:,k].max()) if k<len(Ls)-1 else None))
    summary=dict(name=name,reference_converged=reference_ok,reference_score_RMS=scale,
        reference_direction_RMS=float(rms(reference[:n]-np.asarray(target_score(jnp.asarray(policy[:,None],dtype=jnp.float32))).ravel())),
        strict_1pct_L=stable_minimum(Ls,errors,deltas,.01) if reference_ok else None,
        practical_5pct_L=stable_minimum(Ls,errors,deltas,.05) if reference_ok else None,
        grid_strict_1pct_L=stable_minimum(Ls,grid_errors,rms(np.diff(scores[:,:,n:],axis=1),axis=2)/(gridscale+1e-12),.01) if reference_ok else None,
        largest_L=max(Ls),float32_vs_float64_relative_RMSE=precision_error,results=results)
    np.savez_compressed(out/'scores.npz',Ls=Ls,actions=actions,reference=reference,reference_logp=reference_logp,
        scores=scores,log_density=logps,policy_errors=errors,grid_errors=grid_errors,doubling_changes=deltas)
    write(out/'SUMMARY.json',summary)
    assert sha(checkpoint)==digest,'Parent checkpoint changed'
    assert hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()==parameter_digest,'Actor mutated'
    write(out/'COMPLETE.json',dict(finished=time.time(),seconds=time.time()-metadata['started'],analysis_commit=manifest['commit'],checkpoint_sha256=digest))
    write(out/'STATUS.json',dict(phase='complete',time=time.time()))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
