"""Refine only unresolved quadrature; reuse every original MC score unchanged."""
import argparse,json,os,shutil,time,hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr
from .measure import make_accumulator,quadrature_nodes,stable_minimum,rms,write,sha
from experiments.kl_direction_1d.core import Experiment,target_score

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--index',type=int,required=True)
    a=ap.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    pcfg=json.loads((a.parent/'source/experiments/kl_direction_1d/config.json').read_text())
    cond=pcfg['conditions'][a.index%5];seed=pcfg['seeds'][a.index//5];name=f"{cond['method']}_L{cond['L']}_s{seed}"
    old=a.root/'runtime/results'/name;out=a.root/'runtime/refined_results'/name
    assert (old/'COMPLETE.json').exists();out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    q=json.loads((old/'QUADRATURE.json').read_text())
    if q['converged']:
        for p in old.iterdir():
            if p.is_file():shutil.copy2(p,out/p.name)
        write(out/'REFINEMENT.json',dict(required=False,reason='Original reference passed unchanged criteria'))
        return
    assert jax.default_backend()=='gpu'
    started=time.time();exp=Experiment(pcfg,cond['method'],cond['L'],seed)
    checkpoint=a.parent/'runtime/runs'/name/'checkpoint.msgpack';digest=sha(checkpoint);exp.restore(checkpoint)
    parameter_hash=hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()
    d=dict(np.load(old/'scores.npz'));actions=d['actions'];previous=d['reference'];n=cfg['policy_probes'];count=len(actions)
    acc=make_accumulator(cfg['kernel_chunk']);zeros=(np.full(count,-np.inf),np.zeros(count))
    def components(params,z):
        mu,ls=jax.lax.map(lambda x:exp.components(params,x),z.reshape(-1,4096,1))
        return mu.reshape(-1,1),ls.reshape(-1,1)
    bank=jax.jit(components);stable=0
    for r in q['levels'][::-1]:
        if r.get('passed_refinement'):stable+=1
        else:break
    for panels in [16384,32768,65536,131072,262144,524288]:
        z,w=quadrature_nodes(panels,16,cfg['latent_limit']);mu,ls=bank(exp.state.params,jnp.asarray(z))
        assert mu.dtype==jnp.float32
        lp,s=acc(mu,ls,w,actions,0,len(z)//cfg['kernel_chunk'],zeros);lp,s=np.asarray(lp),np.asarray(s)
        tail=2*ndtr(-cfg['latent_limit']);kmax=2/(np.exp(-5)*np.sqrt(2*np.pi))
        with np.errstate(over='ignore',divide='ignore'):
            bound=(tail*kmax*2*np.exp(10)+np.abs(s)*tail*kmax)/np.maximum(np.exp(lp)-tail*kmax,0)
        diff=s-previous;good=bool(np.all(np.abs(diff)<=cfg['reference_atol']+cfg['reference_rtol']*np.abs(s)) and np.all(bound<1e-5))
        stable=stable+1 if good else 0
        rec=dict(nodes=len(z),max_tail_score_bound=float(bound.max()),max_score_change=float(np.max(np.abs(diff))),policy_relative_RMSE=float(rms(diff[:n])/(rms(s[:n])+1e-12)),passed_refinement=good)
        q['levels'].append(rec);print(json.dumps(dict(name=name,**rec)),flush=True)
        previous=s
        if stable>=2:q['converged']=True;break
    scores=d['scores'];Ls=d['Ls'];scale=float(rms(s[:n]));gscale=float(rms(s[n:]))
    errors=rms(scores[:,:,:n]-s[None,None,:n],axis=2)/(scale+1e-12)
    grids=rms(scores[:,:,n:]-s[None,None,n:],axis=2)/(gscale+1e-12)
    deltas=rms(np.diff(scores[:,:,:n],axis=1),axis=2)/(scale+1e-12)
    summary=json.loads((old/'SUMMARY.json').read_text());oldscale=summary['reference_score_RMS']
    summary.update(reference_converged=q['converged'],reference_score_RMS=scale,
        reference_direction_RMS=float(rms(s[:n]-np.asarray(target_score(jnp.asarray(actions[:n,None],dtype=jnp.float32))).ravel())),
        strict_1pct_L=stable_minimum(Ls,errors,deltas,.01) if q['converged'] else None,
        practical_5pct_L=stable_minimum(Ls,errors,deltas,.05) if q['converged'] else None,
        grid_strict_1pct_L=stable_minimum(Ls,grids,rms(np.diff(scores[:,:,n:],axis=1),axis=2)/(gscale+1e-12),.01) if q['converged'] else None,
        float32_vs_float64_relative_RMSE=summary['float32_vs_float64_relative_RMSE']*oldscale/scale)
    for k,row in enumerate(summary['results']):
        row.update(relative_RMSE_mean=float(errors[:,k].mean()),relative_RMSE_max=float(errors[:,k].max()),absolute_RMSE_mean=float(errors[:,k].mean()*scale),
            pointwise_abs_error_p99=float(np.quantile(np.abs(scores[:,k,:n]-s[:n]),.99)),empirical_bias_RMSE=float(rms(scores[:,k,:n].mean(0)-s[:n])),
            grid_relative_RMSE_max=float(grids[:,k].max()),next_doubling_relative_change_max=float(deltas[:,k].max()) if k<len(Ls)-1 else None)
    d.update(reference=s,reference_logp=lp,policy_errors=errors,grid_errors=grids,doubling_changes=deltas)
    np.savez_compressed(out/'scores.npz',**d);np.savez_compressed(out/'reference.npz',actions=actions,score=s,log_density=lp,tail_score_error_bound=bound)
    for f in ['RUN.json','fixed_actions.npy']:shutil.copy2(old/f,out/f)
    write(out/'SUMMARY.json',summary);write(out/'QUADRATURE.json',q)
    assert sha(checkpoint)==digest
    assert hashlib.sha256(__import__('flax').serialization.to_bytes(exp.state.params)).hexdigest()==parameter_hash
    manifest=json.loads((a.root/'REFINEMENT_SOURCE_MANIFEST.json').read_text())
    receipt=dict(required=True,commit=manifest['commit'],original_scores_sha256=sha(old/'scores.npz'),checkpoint_sha256=digest,
        MC_scores_unchanged=bool(np.array_equal(d['scores'],np.load(old/'scores.npz')['scores'])),reference_converged=q['converged'],seconds=time.time()-started)
    write(out/'REFINEMENT.json',receipt);write(out/'COMPLETE.json',dict(finished=time.time(),**receipt))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
