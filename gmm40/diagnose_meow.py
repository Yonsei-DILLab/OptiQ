"""Read-only energy regression and saturation diagnostics for a saved MEow flow."""
import argparse
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .target import Target,RESULTS
from .torch_agents import MEow
from .evaluation import atomic_json,background


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
    folder=RESULTS/'meow_seed0';checkpoint=folder/'checkpoints'/f'step_{args.step:07d}.bin'
    out=RESULTS/'diagnostics'/f'meow_{args.step:07d}';out.mkdir(parents=True,exist_ok=False)
    target=Target();agent=MEow(target,0,256);agent.restore(checkpoint);agent.actor.eval()
    policy=np.load(folder/'evaluations'/f'step_{args.step:07d}'/'samples.npy')
    reference=target.sample(10000,20260917,bounded=True)
    rng=np.random.default_rng(723);uniform=rng.uniform(-39.99,39.99,size=(10000,2)).astype(np.float32)
    @torch.no_grad()
    def predicted(points):
        results=[]
        for start in range(0,len(points),2048):
            a=torch.tensor(points[start:start+2048]/40,device='cuda',dtype=torch.float32)
            obs=torch.zeros((2*len(a),1),device='cuda')
            q,v=agent.actor.get_qv(obs,torch.cat([a,a]))
            q1,q2=q[:len(a),0]-2*math.log(40),q[len(a):,0]-2*math.log(40)
            logp=agent.actor.log_prob(obs,torch.cat([a,a]))[:len(a)]-2*math.log(40)
            identity_error=torch.max(torch.abs(q1-v[:len(a),0]-logp))
            assert float(identity_error)<2e-4
            results.append(torch.stack([q1,q2,logp],dim=-1).cpu().numpy())
        return np.concatenate(results)
    metrics={}
    for name,points in [('policy',policy),('uniform',uniform),('target_reference',reference)]:
        pred=predicted(points);truth=target.log_prob(points);error=pred[:,:2].mean(-1)-truth
        metrics[name]=dict(Q_rmse=float(np.sqrt(np.mean(error**2))),Q_bias=float(error.mean()),
                           true_Q_mean=float(truth.mean()),model_Q_mean=float(pred[:,:2].mean()),
                           Q_error_quantiles=np.quantile(error,[0,.1,.5,.9,1]).tolist(),
                           near_box_boundary_fraction=float((np.abs(points).max(-1)>39.9).mean()),
                           inverse_clip_region_fraction=float((np.abs(points).max(-1)>=40*(1-1e-5)).mean()),
                           exact_float_boundary_fraction=float((np.abs(points).max(-1)>=40).mean()))
        np.savez_compressed(out/f'{name}_Q.npz',points=points,Q_true=truth,Q_heads=pred[:,:2],analytic_logp=pred[:,2])
    # Keep the pre-tanh output to expose saturation that inverse clipping can hide.
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        torch.manual_seed(900000);torch.cuda.manual_seed_all(900000)
        obs=torch.zeros((10000,1),device='cuda');z,_=agent.actor.prior.sample(10000,context=obs)
        for flow in agent.actor.flows[:-1]:z,_=flow.forward(z,context=obs)
        pre=z.cpu().numpy()
    metrics['pretanh']=dict(abs_max=float(np.abs(pre).max()),abs_coordinate_quantiles=np.quantile(np.abs(pre),[.5,.9,.99,1]).tolist())
    grid=np.linspace(-39.99,39.99,161);xx,yy=np.meshgrid(grid,grid);points=np.c_[xx.ravel(),yy.ravel()].astype(np.float32)
    truth=target.log_prob(points).reshape(xx.shape);pred=predicted(points)[:,:2].mean(-1).reshape(xx.shape)
    np.savez_compressed(out/'Q_grid.npz',x=xx,y=yy,Q_true=truth,Q_model=pred)
    shared_min=min(float(truth.min()),float(pred.min()))
    shared_max=max(float(truth.max()),float(pred.max()))
    residual_limit=float(np.abs(pred-truth).max())
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for ax,values,title in zip(axes,[truth,pred,pred-truth],['Oracle Q = log GMM40','Learned flow Q (head mean)','Learned Q minus oracle Q']):
        bounds=dict(vmin=shared_min,vmax=shared_max) if ax is not axes[2] else dict(vmin=-residual_limit,vmax=residual_limit)
        image=ax.pcolormesh(xx,yy,values,shading='auto',cmap='viridis' if ax is not axes[2] else 'coolwarm',**bounds)
        fig.colorbar(image,ax=ax);ax.scatter(*target.means.T,c='black',s=8,marker='+');ax.set(title=title,aspect='equal',xlabel='x',ylabel='y')
    fig.suptitle(f'MEow | {args.step:,} updates | fixed Q regression diagnostic')
    fig.savefig(out/'Q_comparison.png',dpi=140);fig.savefig(out/'Q_comparison.pdf');plt.close(fig)
    atomic_json(out/'metrics.json',dict(checkpoint=str(checkpoint),updates=args.step,training_performed=False,metrics=metrics,
        caveat='Direct log_prob is evaluated through the native inverse, which clips actions near the boundary; it is not an exact density of floating-point boundary atoms. No KL estimate is asserted.'))
    print(metrics)


if __name__=='__main__':main()
