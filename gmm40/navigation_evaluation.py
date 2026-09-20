"""Actual 100-step environment rollouts, terminal distributions and learned values."""
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon
from .target import Target
from .navigation import vector_rollout
from .evaluation import metrics,atomic_json,background


def evaluate_navigation(folder,name,updates,env_steps,act,q_function,episodes=1000,info=None,temperature=None):
    folder=Path(folder);out=folder/"evaluations"/f"update_{updates:07d}"
    out.mkdir(parents=True,exist_ok=False)
    target=Target()
    rollout=vector_rollout(act,n=episodes,seed=20260918)
    np.savez_compressed(out/"environment_rollout.npz",**rollout)
    terminal=rollout['positions'][-1]
    reference=target.sample(episodes,20260917,bounded=False)
    result=metrics(terminal,target,reference,reference)
    result['outside_target_box_40_fraction']=result.pop('outside_fraction')
    result['near_target_box_40_edge_fraction']=result.pop('boundary_fraction')
    result['histogram_js_box_40']=result.pop('histogram_js')
    edges=np.linspace(-50,50,101)
    hx=np.histogram2d(*terminal.T,bins=(edges,edges))[0].ravel()+1e-8
    hy=np.histogram2d(*reference.T,bins=(edges,edges))[0].ravel()+1e-8
    result['histogram_js']=float(jensenshannon(hx,hy)**2)
    result.update(updates=updates,env_steps=env_steps,mean_return=float(rollout['rewards'].sum(0).mean()),
                  return_std=float(rollout['rewards'].sum(0).std()),terminal_mean_reward=float(rollout['rewards'][-1].mean()),
                  boundary_50_fraction=float((np.abs(terminal).max(1)>49.9).mean()),
                  target_reference="original unbounded DiKL GMM40",training=info or {},policy_temperature=temperature)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for ax in axes:background(ax,target);ax.set(xlim=(-50,50),ylim=(-50,50))
    axes[0].scatter(*reference.T,s=3,alpha=.3,c="#3676b9");axes[0].set_title("Ground truth | original GMM40")
    axes[1].scatter(*terminal.T,s=4,alpha=.35,c="#dd7932");axes[1].set_title(f"{name} | {updates:,} updates\n100-step terminal positions | {result['mode_coverage']}/40 components")
    for i in range(min(64,episodes)):
        path=rollout['positions'][:,i];axes[2].plot(*path.T,linewidth=.7,alpha=.5)
    axes[2].set_title(f"Actual 100-step trajectories\nMMD²={result['mmd2']:.5f}, return={result['mean_return']:.1f}")
    fig.savefig(out/"terminal_and_paths.png",dpi=140);fig.savefig(out/"terminal_and_paths.pdf");plt.close(fig)
    frames=[]
    for t in range(0,101,4):
        fig,ax=plt.subplots(figsize=(5.4,5.4),constrained_layout=True);background(ax,target)
        ax.set(xlim=(-50,50),ylim=(-50,50),title=f"{name} | update {updates:,}\nEnvironment step {t}/100")
        for i in range(min(64,episodes)):
            ax.plot(*rollout['positions'][:t+1,i].T,lw=.6,alpha=.35,color=plt.cm.turbo(i/64))
        ax.scatter(*rollout['positions'][t,:64].T,s=12,c=np.arange(min(64,episodes)),cmap="turbo")
        fig.canvas.draw();frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy());plt.close(fig)
    imageio.mimsave(out/"environment_rollout.gif",frames,duration=120,loop=0)
    if q_function is not None:
        result['q_probe_reference']=json.loads((folder/'config.json').read_text()).get('q_probe_reference','legacy native conservative live Q')
        grid=np.linspace(-50,50,41);xx,yy=np.meshgrid(grid,grid);states=np.stack([xx.ravel(),yy.ravel()],axis=-1).astype(np.float32)
        rng=np.random.default_rng(790);actions=rng.uniform(-1,1,size=(64,2)).astype(np.float32)
        obs=np.repeat(states,64,axis=0);act_grid=np.tile(actions,(len(states),1))
        qs=np.concatenate([np.asarray(q_function(obs[i:i+4096],act_grid[i:i+4096])).reshape(-1) for i in range(0,len(obs),4096)]).reshape(len(states),64)
        plot_temperature=1. if temperature is None else temperature
        logz=(logsumexp(qs/plot_temperature,axis=1)-np.log(64)+np.log(4)).reshape(xx.shape)
        logz_t1=(logsumexp(qs,axis=1)-np.log(64)+np.log(4)).reshape(xx.shape)
        # Uniform-action probes describe the Q landscape, not the actor's proposal ESS.
        result['q_probe']={'states':'fixed 41x41 spatial grid','actions':'64 shared uniform actions',
                           'Q_reference':result['q_probe_reference'],
                           'q_min':float(qs.min()),'q_max':float(qs.max()),
                           'mean_action_q_std':float(qs.std(1).mean()),
                           'mean_action_q_range':float(np.ptp(qs,axis=1).mean()),
                           'uniform_proposal_ess_by_temperature':{}}
        result['partition_estimator']=dict(method='uniform-action Monte Carlo',actions_per_state=64,
                                           action_box_area=4.,temperature=plot_temperature,
                                           caveat='Finite-sample log estimate; not an exact integral. Sharp Q/T landscapes can have large sampling error.')
        for temp in sorted(set([.1,.25,1.,plot_temperature])):
            weights=np.exp(qs/temp-logsumexp(qs/temp,axis=1,keepdims=True))
            result['q_probe']['uniform_proposal_ess_by_temperature'][str(temp)]=float((1/(weights**2).sum(1)).mean())
        np.savez_compressed(out/"learned_log_partition.npz",x=xx,y=yy,log_z=logz,log_z_T1=logz_t1,Q=qs,actions=actions,temperature=plot_temperature,Q_reference=result['q_probe_reference'])
        fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
        im=ax.pcolormesh(xx,yy,logz,shading="auto",cmap="viridis");fig.colorbar(im,ax=ax,label=f"log ∫ exp(Q(s,a)/T) da | T={plot_temperature:.4g}")
        ax.contour(xx,yy,target.log_prob(np.stack([xx,yy],axis=-1)),levels=[-12,-9,-7,-6],colors="white",linewidths=.5)
        ax.set(title=f"{name} | learned state log partition\n64-action Monte Carlo estimate",xlabel="x",ylabel="y",aspect="equal")
        fig.savefig(out/"learned_log_partition.png",dpi=140);plt.close(fig)
    atomic_json(out/"metrics.json",result)
    with (folder/"metrics.jsonl").open("a") as f:f.write(json.dumps(result,allow_nan=False)+"\n")
    atomic_json(folder/"latest.json",result)
    images=sorted((folder/"evaluations").glob("*/terminal_and_paths.png"))
    imageio.mimsave(folder/"training_evolution.gif",[imageio.imread(p) for p in images],duration=1000,loop=0)
    print(json.dumps(dict(event="navigation_eval",method=name,updates=updates,env_steps=env_steps,mmd2=result['mmd2'],modes=result['mode_coverage'],mean_return=result['mean_return'])),flush=True)
    if not name.startswith('validation_'):
        from .report import refresh
        refresh()
    return result
