"""Fixed-anchor OT targets under resampled peer latents, without training."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import atomic_json
from .optiq import OptiQ
from .target import RESULTS, Target
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.transport import sinkhorn


@partial(jax.jit, static_argnames=('epsilon','iterations'))
def anchor_moments(mu, actions, pretanh, weights, epsilon, iterations=1000):
    cost = ((jnp.tanh(mu)[:,:,None,:]-actions[:,None,:,:])**2).sum(-1)
    plan = sinkhorn(cost,weights,epsilon,iterations)
    row = plan[:,0]/jnp.maximum(plan[:,0].sum(-1,keepdims=True),1e-30)
    mean = jnp.einsum('bm,bmd->bd',row,pretanh)
    variance = jnp.maximum(jnp.einsum('bm,bmd->bd',row,pretanh**2)-mean**2,0)
    return mean, variance, mu.shape[1]*jnp.abs(plan.sum(-1)-1/mu.shape[1]).max(-1)


def render_plot(rows,out):
    sizes=list(dict.fromkeys((row['n'],row['m']) for row in rows))
    epsilons=list(dict.fromkeys(row['epsilon'] for row in rows))
    fig,axes=plt.subplots(len(epsilons),len(sizes),figsize=(5*len(sizes),4*len(epsilons)),
                          squeeze=False,sharey=True,constrained_layout=True)
    ceiling=1.12*max(max(row['variance_target_at_current_mu'],row['actor_variance']) for row in rows)
    labels={'teachers_only':'Teacher only','peers_only_counterfactual':'Peers (CF)','peers_and_teachers':'Both'}
    for iy,epsilon in enumerate(epsilons):
        for ix,(n,m) in enumerate(sizes):
            selected=[row for row in rows if (row['n'],row['m'],row['epsilon'])==(n,m,epsilon)]
            ax=axes[iy,ix];x=np.arange(len(selected));bottom=np.zeros(len(selected))
            for field,label,color in [('within_row_variance','Within OT row','#5e81ac'),
                                      ('across_draw_mean_variance','Across repeated draws','#d08770'),
                                      ('mean_fit_bias_squared','Bias at current mean','#a3be8c')]:
                values=np.array([row[field] for row in selected])
                ax.bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
            ax.axhline(selected[0]['actor_variance'],color='black',linestyle='--',label='Current actor variance')
            ax.set(xticks=x,xticklabels=[labels[row['case']] for row in selected],
                   title=f'N={n}, M={m} | OT ε={epsilon}',ylim=(0,ceiling))
            ax.tick_params(axis='x',labelsize=9)
            if ix==0:ax.set_ylabel('Pre-tanh variance per coordinate')
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Fixed-anchor OT supervision | saved actor, no training\nCF holds the teacher cloud fixed; arm differences are not additive contributions.',fontsize=12)
    fig.savefig(out/'anchor_variance.png',dpi=150)
    fig.savefig(out/'anchor_variance.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--name',default='optiq_anchor_competition_100k')
    parser.add_argument('--anchors',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=64)
    parser.add_argument('--sizes',nargs='+',default=['16x64','64x256'])
    parser.add_argument('--epsilons',nargs='+',type=float,default=[.1,.01])
    parser.add_argument('--iterations',type=int,default=1000)
    args=parser.parse_args()
    sizes=[tuple(map(int,value.lower().split('x'))) for value in args.sizes]
    if any(len(size)!=2 or size[0]<1 or size[1]<size[0] or size[1]%size[0] for size in sizes):
        raise ValueError('Each N×M pair must be positive, with M a multiple of N')
    if args.anchors<1 or args.repeats<2 or args.iterations<1 or any(e<=0 for e in args.epsilons):
        raise ValueError('Invalid probe size, epsilon, or iteration count')
    max_n=max(n for n,m in sizes)
    checkpoint=args.checkpoint.resolve()
    original_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg=json.loads((checkpoint.parent.parent/'config.json').read_text())
    target=Target()
    agent=OptiQ(target,hidden_dims=(cfg.get('width',256),)*cfg.get('depth',2))
    agent.restore(checkpoint)
    out=RESULTS/'diagnostics'/args.name
    out.mkdir(parents=True,exist_ok=False)
    a,r=args.anchors,args.repeats
    key=jax.random.PRNGKey(87319)
    anchors=jax.random.normal(key,(a,2))
    z=jax.random.normal(jax.random.fold_in(key,1),(a,r,max_n,2))
    z=z.at[:,:,0,:].set(jnp.broadcast_to(anchors[:,None,:],(a,r,2)))
    all_mu,all_ls=agent.actor.apply({'params':agent.state.params},jnp.zeros((a*r*max_n,1)),z.reshape(-1,2))
    all_mu,all_ls=all_mu.reshape(a,r,max_n,2),all_ls.reshape(a,r,max_n,2)
    anchor_mu=np.asarray(all_mu[:,:,0,:])
    assert np.allclose(anchor_mu,anchor_mu[:,:1],rtol=0,atol=1e-6)
    actor_variance=float(jnp.exp(2*all_ls[:,:,0,:]).mean())
    rows=[]
    controls={}
    for n,m in sizes:
        varying_mu=all_mu[:,:,:n].reshape(a*r,n,2)
        varying_ls=all_ls[:,:,:n].reshape(a*r,n,2)
        fixed_mu=jnp.broadcast_to(all_mu[:,:1,:n],(a,r,n,2)).reshape(a*r,n,2)
        fixed_ls=jnp.broadcast_to(all_ls[:,:1,:n],(a,r,n,2)).reshape(a*r,n,2)
        clouds={}
        for label,mu,ls in [('fixed',fixed_mu,fixed_ls),('varying',varying_mu,varying_ls)]:
            proposal=ConditionalGaussianProposal(mu,ls,.05)
            actions,u,_=proposal.sample(jax.random.fold_in(key,m),m//n,'exact')
            weights=jax.nn.softmax(target.jax_log_prob(40*actions)-proposal.log_prob(u)+2*math.log(40),axis=-1)
            clouds[label]=(actions,u,weights)
        def first_cloud(value):
            shaped=value.reshape((a,r)+value.shape[1:])
            return jnp.broadcast_to(shaped[:,:1],shaped.shape).reshape(value.shape)
        held=tuple(first_cloud(value) for value in clouds['fixed'])
        cases=[('teachers_only',fixed_mu,clouds['fixed']),
               ('peers_only_counterfactual',varying_mu,held),
               ('peers_and_teachers',varying_mu,clouds['varying'])]
        for epsilon in args.epsilons:
            base=None
            for label,mu,(actions,u,weights) in cases:
                mean,var,error=map(np.asarray,anchor_moments(mu,actions,u,weights,epsilon,args.iterations))
                mean,var=mean.reshape(a,r,2),var.reshape(a,r,2)
                if base is None:
                    base=(mean[:,0].copy(),var[:,0].copy())
                controls[f'{n}x{m}_eps{epsilon}_{label}']=bool(np.allclose(mean[:,0],base[0],rtol=0,atol=2e-5) and np.allclose(var[:,0],base[1],rtol=0,atol=2e-5))
                within=float(var.mean())
                between=float(mean.var(1).mean())
                bias=float(((mean.mean(1)-anchor_mu[:,0])**2).mean())
                physical=40*np.tanh(mean)
                distance=(((physical[:,:,None,:]-target.means[None,None,:,:])/target.std[None,None,:,None])**2).sum(-1)
                stats=dict(n=n,m=m,epsilon=epsilon,case=label,
                           within_row_variance=within,across_draw_mean_variance=between,
                           mean_fit_bias_squared=bias,
                           best_variance_if_mean_can_fit=within+between,
                           variance_target_at_current_mu=within+between+bias,
                           actor_variance=actor_variance,
                           anchor_barycenter_precision=float((distance.min(-1)<=9).mean()),
                           relative_row_error_max=float(error.max()),
                           relative_row_error_mean=float(error.mean()),
                           teacher_ess=float((1/(weights**2).sum(-1)).mean()))
                assert all(np.isfinite(value) for value in stats.values() if isinstance(value,(float,int)))
                rows.append(stats)
                np.savez_compressed(out/f'n{n}_m{m}_eps{epsilon}_{label}.npz',anchor_latents=np.asarray(anchors),
                                    anchor_mu=anchor_mu[:,0],row_mean=mean,row_variance=var,relative_row_error=error)
                print(json.dumps(stats),flush=True)
    if not all(controls.values()):
        raise RuntimeError(f'Common first-draw control failed: {controls}')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=original_hash:
        raise RuntimeError('Input checkpoint changed')
    atomic_json(out/'results.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=original_hash,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                updates=agent.updates,temperature=1,anchors=a,repeats=r,sizes=sizes,iterations=args.iterations,training_performed=False,
                common_first_draw_checks=controls,input_checkpoint_unchanged=True,rows=rows,
                caveat='Peer-only arm holds one teacher cloud per anchor and its generating density fixed: it is a counterfactual isolation, not the native algorithm. Variance differences between arms are not an additive causal decomposition. All conclusions are conditional on this saved actor and finite anchor draws.'))
    short={'teachers_only':'teacher','peers_only_counterfactual':'peers (CF)','peers_and_teachers':'both'}
    render_plot(rows,out)
    lines=['# Fixed-anchor student competition diagnostic','',
           'T=1; saved actor only. No optimizer step is taken. Each anchor latent remains exactly fixed.',
           f'{a} anchor latents × {r} draws, Sinkhorn {args.iterations:,} iterations.',
           'Teacher-only: fixed peers, fresh teacher clouds. Both: fresh peers and native conditional-proposal teachers.',
           'Peers (CF): fresh peers but one fixed teacher cloud per anchor, retaining its generating proposal density.',
           'The CF arm isolates a perturbation; it is not the native training distribution or an additive contribution estimate.',
           'The first draw is shared across all three arms and was checked numerically.', '',
           '| N×M | ε | Draws changed | Within row | Across draws | Bias² | Variance at current μ | Barycenter precision | Max relative row error |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['n']}×{row['m']} | {row['epsilon']} | {short[row['case']]} | {row['within_row_variance']:.4f} | {row['across_draw_mean_variance']:.4f} | {row['mean_fit_bias_squared']:.4f} | {row['variance_target_at_current_mu']:.4f} | {row['anchor_barycenter_precision']:.1%} | {row['relative_row_error_max']:.4g} |")
    lines.extend(['','With a freely fitted conditional mean, the variance target is within-row plus across-draw variance.',
                  'Keeping the present actor mean adds the squared mean-fitting bias. These local targets do not establish retraining performance.',
                  '', '![Anchor target variance](anchor_variance.png)'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
