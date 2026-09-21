"""Read-only Monte Carlo analysis of fixed-Q OT supervision at a saved actor.

No optimizer step is taken. Repeated teacher clouds at fixed student latents
separate row variance from variation of the supervised mean across clouds.
"""
import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .target import Target,RESULTS
from .optiq import OptiQ
from .evaluation import atomic_json
from gmm40._v5.optiq_dime.semi_implicit import ConditionalGaussianProposal
from gmm40._v5.optiq_dime.transport import sinkhorn


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--name',default='optiq_100k_supervision_diagnostic')
    args=parser.parse_args()
    config=json.loads((args.checkpoint.parent.parent/'config.json').read_text())
    target=Target();agent=OptiQ(target,hidden_dims=(config.get('width',256),)*config.get('depth',2));agent.restore(args.checkpoint)
    out=RESULTS/'diagnostics'/args.name;out.mkdir(parents=True,exist_ok=False)
    key=jax.random.PRNGKey(87063)
    repeats=128
    means=jnp.asarray(target.means);std=jnp.asarray(target.std)
    results=[]
    for n in (16,64):
        # Prefix-identical latent clouds make N=16 versus N=64 easier to assess.
        z=jax.random.normal(key,(64,2))[:n]
        mu,ls=agent.actor.apply({'params':agent.state.params},jnp.zeros((n,1)),z)
        mu=jnp.broadcast_to(mu,(repeats,n,2));ls=jnp.broadcast_to(ls,mu.shape)
        proposal=ConditionalGaussianProposal(mu,ls,.05)
        for m in ((64,256,1024) if n==16 else (256,1024)):
            a,u,_=proposal.sample(jax.random.fold_in(key,m),m//n,'exact')
            q=target.jax_log_prob(40*a)
            logq=proposal.log_prob(u)-2*math.log(40)
            w=jax.nn.softmax(q-logq,axis=-1)
            distance=jnp.sum(((40*a[:,:,None,:]-means[None,None,:,:])/std[None,None,:,None])**2,axis=-1)
            near=jnp.min(distance,axis=-1)<=9
            component_labels=jax.nn.one_hot(jnp.argmin(distance,axis=-1),40)
            cost=jnp.sum((jnp.tanh(mu)[:,:,None,:]-a[:,None,:,:])**2,axis=-1)
            for epsilon in (.1,.03,.01):
                @jax.jit
                def analyze(cost,w,u,mu,ls):
                    # Use more iterations for the diagnostic so smoothing is not confused with solver error.
                    plan=sinkhorn(cost,w,epsilon,1000)
                    row=plan/jnp.maximum(plan.sum(-1,keepdims=True),1e-30)
                    row_mean=jnp.einsum('bnm,bmd->bnd',row,u)
                    row_var=jnp.maximum(jnp.einsum('bnm,bmd->bnd',row,u*u)-row_mean**2,0)
                    component_mass=jnp.einsum('bnm,bmc->bnc',row,component_labels)
                    component_entropy=-(component_mass*jnp.log(jnp.maximum(component_mass,1e-30))).sum(-1)
                    barycenter=40*jnp.tanh(row_mean)
                    barycenter_distance=jnp.sum(((barycenter[:,:,None,:]-means[None,None,:,:])/std[None,None,:,None])**2,axis=-1)
                    expected_mean=row_mean.mean(0)
                    within=row_var.mean()
                    between=row_mean.var(0).mean()
                    bias=((mu[0]-expected_mean)**2).mean()
                    return dict(within_row_variance=within,between_cloud_mean_variance=between,
                                squared_mean_bias=bias,nll_optimal_variance=within+between+bias,
                                actor_variance=jnp.exp(2*ls).mean(),
                                row_effective_components=jnp.exp(component_entropy).mean(),
                                row_largest_component_mass=component_mass.max(-1).mean(),
                                row_barycenter_precision=(barycenter_distance.min(-1)<=9).mean(),
                                row_entropy=-(row*jnp.log(jnp.maximum(row,1e-30))).sum(-1).mean(),
                                row_error_max=jnp.abs(plan.sum(-1)-1/n).max(),
                                row_error_relative_max=n*jnp.abs(plan.sum(-1)-1/n).max(),
                                column_error_max=jnp.abs(plan.sum(-2)-w).max())
                stats={k:float(v) for k,v in analyze(cost,w,u,mu,ls).items()}
                stats.update(n=n,m=m,epsilon=epsilon,iterations=1000,teacher_ess=float((1/(w*w).sum(-1)).mean()),
                             teacher_weighted_precision=float((w*near).sum(-1).mean()),
                             teacher_unweighted_precision=float(near.mean()),teacher_weighted_Q=float((w*q).sum(-1).mean()))
                assert all(np.isfinite(v) for v in stats.values())
                results.append(stats)
                print(json.dumps(stats),flush=True)
    atomic_json(out/'results.json',dict(checkpoint=str(args.checkpoint),updates=agent.updates,
                temperature=1,teacher_repetitions=repeats,training_performed=False,
                interpretation='Per-coordinate pre-tanh variance; fixed student latents; fresh actor-proposal teachers. Equal latent weights.',results=results))
    x=np.arange(len(results));fig,ax=plt.subplots(figsize=(13,6),constrained_layout=True)
    lower=np.zeros(len(results))
    for key,label,color in [('within_row_variance','Within OT row','#5e81ac'),('between_cloud_mean_variance','Mean varies across teacher clouds','#d08770'),('squared_mean_bias','Mean fitting bias','#a3be8c')]:
        values=np.array([r[key] for r in results]);ax.bar(x,values,bottom=lower,label=label,color=color);lower+=values
    ax.plot(x,[r['actor_variance'] for r in results],'k--',label='Current actor variance')
    ax.set(xticks=x,xticklabels=[f"{r['n']}×{r['m']}\nε={r['epsilon']}" for r in results],ylabel='Pre-tanh variance per coordinate',title='Read-only decomposition of OptiQ NLL supervision at 100K')
    ax.legend();fig.savefig(out/'variance_decomposition.png',dpi=150);fig.savefig(out/'variance_decomposition.pdf');plt.close(fig)
    lines=['# OptiQ supervision diagnostic','',
           'Read-only at the saved 100K actor; **no training or target samples used**. T=1 throughout.',
           'Student latent points are fixed. Repeat actor-proposal sampling 128 times and solve OT with 1,000 iterations.',
           'Expected NLL variance = within-row variance + variance of the row mean across teacher draws + squared mean bias.',
           'This diagnoses local supervision; it does not establish final performance of any retrained variant.',
           '', '| N | M | ε | ESS | Weighted precision | Within row | Across clouds | Bias² | Optimal σ² | Relative max row error |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['n']} | {r['m']} | {r['epsilon']} | {r['teacher_ess']:.2f} | {r['teacher_weighted_precision']:.3f} | {r['within_row_variance']:.4f} | {r['between_cloud_mean_variance']:.4f} | {r['squared_mean_bias']:.4f} | {r['nll_optimal_variance']:.4f} | {r['row_error_relative_max']:.4g} |")
    lines.extend(['','## Conditional row shape','',
                  'Nearest target-component labels are used only for this read-only diagnostic, never in training.',
                  'Barycenters are 40*tanh(E_R[u]); their precision uses the same nearest-center 3σ rule.',
                  '', '| N | M | ε | Effective components per row | Largest component mass | Barycenter precision |',
                  '|---:|---:|---:|---:|---:|---:|'])
    for r in results:lines.append(f"| {r['n']} | {r['m']} | {r['epsilon']} | {r['row_effective_components']:.3f} | {r['row_largest_component_mass']:.3f} | {r['row_barycenter_precision']:.3f} |")
    lines.extend(['','![Variance decomposition](variance_decomposition.png)'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
