"""Common, evaluation-only reference samples, metrics and incremental artifacts."""
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from .target import RESULTS, SCALE


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def mmd2(x, y, limit=2048):
    x,y = x[:limit].astype(np.float64), y[:limit].astype(np.float64)
    xx, yy, xy = cdist(x,x,"sqeuclidean"), cdist(y,y,"sqeuclidean"), cdist(x,y,"sqeuclidean")
    scores={}
    for h in (1.,2.,5.,10.,20.):
        kxx,kyy,kxy=np.exp(-xx/(2*h*h)),np.exp(-yy/(2*h*h)),np.exp(-xy/(2*h*h))
        scores[str(h)] = float((kxx.sum()-len(x))/(len(x)*(len(x)-1))+(kyy.sum()-len(y))/(len(y)*(len(y)-1))-2*kxy.mean())
    return float(np.mean(list(scores.values()))),scores


def assignments(x,target):
    dist=cdist(x,target.means,"sqeuclidean")/(target.std[None,:]**2)
    labels=dist.argmin(axis=1)
    near=dist[np.arange(len(x)),labels] <= 9.0
    counts=np.bincount(labels[near],minlength=40)
    return labels,near,counts


def metrics(x,target,reference,full_reference):
    assert np.isfinite(x).all() and x.shape[1]==2
    labels,near,counts=assignments(x,target)
    _,_,refcounts=assignments(reference,target)
    threshold=np.maximum(10,0.1*refcounts*len(x)/len(reference))
    covered=counts>=threshold
    probs=counts/max(counts.sum(),1)
    refprobs=refcounts/refcounts.sum()
    rng=np.random.default_rng(451)
    directions=rng.normal(size=(128,2)); directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    sw2=np.sqrt(np.mean((np.sort(x@directions.T,axis=0)-np.sort(reference@directions.T,axis=0))**2))
    mmd,by_bandwidth=mmd2(x,reference)
    full_mmd,_=mmd2(x,full_reference)
    edges=np.linspace(-40,40,81)
    hx=np.histogram2d(*x.T,bins=(edges,edges))[0].ravel()+1e-8
    hy=np.histogram2d(*reference.T,bins=(edges,edges))[0].ravel()+1e-8
    return dict(mode_coverage=int(covered.sum()),coverage_reference='40 GMM component centers; not an exact count of density local maxima',covered_modes=np.flatnonzero(covered).tolist(),
                mode_counts_3sigma=counts.tolist(), coverage_threshold=threshold.tolist(),
                high_density_fraction=float(near.mean()), mode_mass_tv=float(0.5*np.abs(probs-refprobs).sum()),
                mode_mass_js=float(jensenshannon(probs+1e-12,refprobs+1e-12)**2),
                effective_modes=float(np.exp(-np.sum(probs*np.log(probs+1e-30)))),
                mmd2=mmd,mmd2_by_bandwidth=by_bandwidth,mmd2_full_unbounded_target=full_mmd,
                sliced_wasserstein2=float(sw2), histogram_js=float(jensenshannon(hx,hy)**2),
                mean_log_target=float(target.log_prob(x).mean()),
                boundary_fraction=float((np.abs(x).max(axis=1)>39.9).mean()),
                outside_fraction=float(np.any(np.abs(x)>40,axis=1).mean()),
                n_samples=len(x),mean=x.mean(0).tolist(),std=x.std(0).tolist())


def background(ax,target):
    grid=np.linspace(-40,40,240)
    xx,yy=np.meshgrid(grid,grid)
    lp=target.log_prob(np.stack([xx,yy],axis=-1))
    ax.contour(xx,yy,lp,levels=[-12,-9,-7,-6,-5],colors="0.65",linewidths=.55)
    ax.scatter(*target.means.T,s=12,c="black",marker="+",label="Target component centers")
    ax.set(xlim=(-40,40),ylim=(-40,40),xlabel="x",ylabel="y",aspect="equal")


def save_evaluation(folder,name,step,samples,target,reference,full_reference,training_info,rollout=None,extra_samples=None):
    folder=Path(folder); destination=folder/"evaluations"/f"step_{step:07d}"
    destination.mkdir(parents=True,exist_ok=False)
    np.save(destination/"samples.npy",samples)
    result=metrics(samples,target,reference,full_reference)
    result.update(method=name,step=step,training=training_info)
    atomic_json(destination/"metrics.json",result)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    for ax in axes[:2]: background(ax,target)
    axes[0].scatter(*reference[:5000].T,s=2,alpha=.25,color="#3676b9")
    axes[0].set_title("Ground truth | bounded GMM40")
    axes[1].scatter(*samples[:5000].T,s=2,alpha=.35,color="#dd7932")
    axes[1].set_title(f"{name} | update {step:,}\n{result['mode_coverage']}/40 components, MMD² {result['mmd2']:.4g}")
    _,_,rc=assignments(reference,target)
    axes[2].bar(np.arange(40)-.18,rc/len(reference),width=.36,label="Target",color="#3676b9")
    axes[2].bar(np.arange(40)+.18,np.asarray(result['mode_counts_3sigma'])/len(samples),width=.36,label=name,color="#dd7932")
    axes[2].set(xlabel="Nearest component (within 3σ)",ylabel="Sample mass",title=f"SW₂ {result['sliced_wasserstein2']:.3f} | precision {result['high_density_fraction']:.1%}")
    axes[2].legend()
    fig.savefig(destination/"samples.png",dpi=140)
    fig.savefig(destination/"samples.pdf")
    plt.close(fig)
    if extra_samples:
        for label,values in extra_samples.items():
            np.save(destination/f"samples_{label}.npy",values)
            atomic_json(destination/f"metrics_{label}.json",metrics(values,target,reference,full_reference))
    if rollout is not None:
        np.savez_compressed(destination/"generation_rollout.npz",positions=rollout)
        make_rollout(destination/"generation_rollout.gif",rollout,target,name)
    with (folder/"metrics.jsonl").open("a") as f: f.write(json.dumps(result,allow_nan=False)+"\n")
    atomic_json(folder/"latest.json",result)
    images=sorted((folder/"evaluations").glob("*/samples.png"))
    imageio.mimsave(folder/"training_evolution.gif",[imageio.imread(p) for p in images],duration=1000,loop=0)
    summary_report(folder,name)
    if not name.startswith('validation_'):
        from .report import refresh
        refresh()
    return result


def make_rollout(path,positions,target,name):
    positions=np.asarray(positions)
    indices=np.unique(np.linspace(0,len(positions)-1,min(25,len(positions))).astype(int))
    frames=[]
    for i in indices:
        fig,ax=plt.subplots(figsize=(5.4,5.4),constrained_layout=True)
        background(ax,target)
        ax.scatter(*positions[i].T,s=8,c=np.arange(positions.shape[1]),cmap="turbo",alpha=.8)
        inside=float((np.abs(positions[i]).max(axis=1)<=40).mean())
        ax.set_title(f"{name} | generation stage {i}/{len(positions)-1}\nModel sampling stages; not environment navigation\n{inside:.0%} of particles inside displayed box")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        plt.close(fig)
    imageio.mimsave(path,frames,duration=120 if len(frames)>2 else 800,loop=0)


def summary_report(folder,name):
    rows=[json.loads(s) for s in (folder/"metrics.jsonl").read_text().splitlines()]
    lines=[f"# {name}: intermediate results", "", "All checkpoints are retained. Reference samples are evaluation-only.","", "| Updates | Components /40 | MMD² ↓ | SW₂ ↓ | High-density fraction ↑ | Component-mass TV ↓ | Training seconds |", "|---:|---:|---:|---:|---:|---:|---:|"]
    config_path=folder/'config.json'
    cfg=json.loads(config_path.read_text()) if config_path.exists() else {}
    if cfg.get('fixed_q_temperature_control'):
        lines[3:3]=[f"Teacher T={cfg['temperature']:g}: the training Boltzmann target is proportional to p_GMM^(1/T). All metrics and blue samples still use the original T=1 bounded GMM for comparison; these are not fit metrics to the tempered target.", '']
    for r in rows:
        lines.append(f"| {r['step']} | {r['mode_coverage']} | {r['mmd2']:.6f} | {r['sliced_wasserstein2']:.3f} | {r['high_density_fraction']:.3f} | {r['mode_mass_tv']:.3f} | {r['training'].get('train_seconds',0):.1f} |")
    lines += ["", "MMD² is the unbiased multi-bandwidth RBF estimate and can be slightly negative.", "Coverage: nearest-component-center distance ≤3σ and count ≥ max(10, 10% of matched reference count). This is not an exact count of density local maxima.", "Component-mass TV conditions on samples within 3σ; inspect high-density fraction alongside it.", "Blue: bounded ground-truth samples. Orange: policy samples. Gray contours/black crosses: true target.", "", "[Training evolution](training_evolution.gif)"]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n")
