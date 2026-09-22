"""Evaluation only: target samples and component identities never enter learners."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def responsibility_mass(x,means,std):
    x=np.asarray(x,np.float64);means=np.asarray(means,np.float64);std=np.asarray(std,np.float64)
    # Equal target mixture weights; log normalization is essential for far-away x.
    logk=-.5*(((x[:,None]-means)/std[:,None])**2).sum(-1)-2*np.log(std)
    r=np.exp(logk-logsumexp(logk,axis=1,keepdims=True))
    return r.mean(0), np.min((((x[:,None]-means)/std[:,None])**2).sum(-1),axis=1)<=9


def sliced_w2(x,y,projections=128):
    if len(x)!=len(y):raise ValueError('Equal evaluation sample counts required')
    rng=np.random.default_rng(451)
    dirs=rng.normal(size=(projections,x.shape[1]));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    # float64 CPU projections; explicit W2 square root, physical action units.
    return float(np.sqrt(np.mean((np.sort(np.asarray(x,np.float64)@dirs.T,axis=0)-np.sort(np.asarray(y,np.float64)@dirs.T,axis=0))**2)))


def metrics(x,target,reference):
    if x.shape!=reference.shape or not np.isfinite(x).all():raise ValueError('Invalid action samples')
    w,near=responsibility_mass(x,target.means,target.std)
    thresholds={str(t):int(np.sum(w>=t/40)) for t in (.1,.25,.5)}
    return dict(coverage=thresholds['0.25']/40,recovered_components=thresholds['0.25'],
        recovered_by_threshold=thresholds,component_mass=w.tolist(),component_mass_tv=float(.5*np.abs(w-1/40).sum()),
        sliced_w2=sliced_w2(x,reference),high_density_fraction=float(near.mean()),
        mean_log_target=float(target.log_prob(x).mean()),eval_samples=len(x))


def sample_actions(agent,n,seed,chunk=4096):
    # Native full stochastic actions, NEVER mu-only. Separate eval RNG in every adapter.
    return np.concatenate([agent.evaluate_samples(min(chunk,n-start),seed+start)[0] for start in range(0,n,chunk)])


def latency(agent,method,warmup=50,repetitions=200,blocks=5):
    """Host-observed single-action sample including RNG and output transfer; warmed.

    No energy calls, teacher construction, CPU plots or density evaluation. GPU is
    synchronized by the one-action host copy on every call. Eval RNG is independent.
    JAX adapters are JIT; PyTorch adapters eager (recorded, not hidden).
    """
    from gmm40.target import SCALE
    if method in ('sac','dipo','meow'):
        import torch
        actor=agent.actor;was_training=actor.training;actor.eval()
        obs=torch.zeros((1,1),device=agent.device)
        def call():
            if method=='sac':out=SCALE*torch.tanh(actor(obs)[0])
            elif method=='dipo':out=SCALE*actor.sample(obs).clamp(-1,1)
            else:out=SCALE*actor.sample(1,obs)[0]
            return out.cpu().numpy()
        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
            torch.manual_seed(771);torch.cuda.manual_seed_all(771)
            for _ in range(warmup):call()
            times=[]
            for _ in range(blocks):
                start=time.perf_counter()
                for _ in range(repetitions):call()
                times.append((time.perf_counter()-start)*1e6/repetitions)
        actor.train(was_training);backend='PyTorch eager'
    else:
        import jax
        import jax.numpy as jnp
        params=agent.state.params
        if method=='sql':
            fn=jax.jit(lambda p,k: SCALE*agent.learner.sample_fn(p,jnp.zeros((1,1)),k))
        else:fn=jax.jit(lambda p,k:agent._sample(p,k,1)[0])
        # Key generation is part of the timed compiled call, and changes each call.
        @jax.jit
        def infer(p,key):
            key,sub=jax.random.split(key)
            return key,fn(p,sub)
        key=jax.random.PRNGKey(771)
        def call():
            nonlocal key
            key,out=infer(params,key)
            return np.asarray(out)
        for _ in range(warmup):call()
        times=[]
        for _ in range(blocks):
            start=time.perf_counter()
            for _ in range(repetitions):call()
            times.append((time.perf_counter()-start)*1e6/repetitions)
        backend='JAX JIT'
    return dict(single_action_us=float(np.median(times)),block_us=times,backend=backend,
        batch=1,warmup=warmup,repetitions=repetitions,blocks=blocks,
        timing='RNG + native full stochastic sampler + synchronized host action transfer; no Q')


def plot_samples(path,samples,target,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    grid=np.linspace(-50,50,250);xx,yy=np.meshgrid(grid,grid)
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    axs[0].contourf(xx,yy,np.exp(target.log_prob(np.stack([xx,yy],-1))),levels=25,cmap='magma')
    axs[0].set_title('Exact target density (evaluation only)')
    axs[1].hist2d(*samples.T,bins=150,range=[[-50,50],[-50,50]],norm=LogNorm(),cmap='magma')
    axs[1].set_title(name+' | 32,768 native action samples')
    mass,_=responsibility_mass(samples,target.means,target.std)
    axs[2].bar(np.arange(40),mass,color='#3676b9')
    axs[2].axhline(1/40,color='black',ls='--',label='Target mass 1/40')
    axs[2].set(xlabel='Target component',ylabel='Posterior responsibility mass',title='Component mass (evaluation only)')
    axs[2].legend(fontsize=8)
    for ax in axs[:2]:ax.set(xlabel='Action 1',ylabel='Action 2',xlim=(-50,50),ylim=(-50,50),aspect='equal')
    fig.savefig(path,dpi=150);plt.close(fig)
