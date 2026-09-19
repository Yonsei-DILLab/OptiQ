"""Separate mode allocation from within-mode fit; all actor plots use raw draws."""
import numpy as np
from scipy.special import logsumexp


def reference(locs,scales,count=32768,seed=20260919):
    rng=np.random.default_rng(seed);labels=rng.integers(0,40,count)
    return locs[labels]+scales[labels]*rng.normal(size=(count,2))


def component_stats(x,locs,scales,weights=None):
    x=np.asarray(x,np.float64);locs=np.asarray(locs,np.float64);scales=np.asarray(scales,np.float64)
    weight=np.ones(len(x))/len(x) if weights is None else np.asarray(weights,np.float64)/np.sum(weights)
    delta=(x[:,None]-locs[None])/scales[None]
    logc=-.5*(delta**2).sum(-1)-np.log(scales).sum(-1)-np.log(2*np.pi)
    resp=np.exp(logc-logsumexp(logc,axis=-1,keepdims=True));wr=weight[:,None]*resp
    mass=wr.sum(0);den=np.maximum(mass,1e-20)
    mean=np.einsum('nk,nd->kd',wr,x)/den[:,None]
    second=np.einsum('nk,nd,ne->kde',wr,x,x)/den[:,None,None]
    cov=second-mean[:,:,None]*mean[:,None,:]
    eigen=np.linalg.eigvalsh(cov/(scales[:,:,None]*scales[:,None,:]))
    core=(delta**2).sum(-1)<=9.
    return dict(mass=mass,mean=mean,cov=cov,eigen=eigen,
        center_error=np.linalg.norm((mean-locs)/scales,axis=-1),
        core_mass=(wr*core).sum(0),logp=logsumexp(logc,axis=1)-np.log(40))


def evaluate(x,ref,locs,scales):
    s=component_stats(x,locs,scales);r=component_stats(ref,locs,scales)
    valid=s['mass']>=.1/40
    rng=np.random.default_rng(718);directions=rng.normal(size=(32,2));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    sw2=np.sqrt(np.mean((np.sort(x@directions.T,axis=0)-np.sort(ref@directions.T,axis=0))**2))
    # Compare sample histograms at a fixed resolution, with independent-reference floor recorded.
    hist=lambda a:np.histogram2d(a[:,0],a[:,1],bins=160,range=[[-50,50],[-50,50]])[0]/len(a)
    hx,hr=hist(x),hist(ref)
    metrics=dict(mode_mass_tv=float(.5*np.abs(s['mass']-1/40).sum()),
        mode_mass_tv_vs_reference=float(.5*np.abs(s['mass']-r['mass']).sum()),
        modes_covered=int((s['core_mass']>=.25/40).sum()),
        coverage_rule='posterior-weighted mass within 3 sigma >= 25% of nominal 1/40',
        within_mode_center_error=float(s['center_error'][valid].mean()) if valid.any() else None,
        within_mode_cov_error=float(np.abs(s['eigen'][valid]-1).mean()) if valid.any() else None,
        within_mode_evaluated_modes=int(valid.sum()),
        mean_logp=float(s['logp'].mean()),reference_mean_logp=float(r['logp'].mean()),
        energy_mean_error=float(abs(s['logp'].mean()-r['logp'].mean())),
        sliced_w2=float(sw2),histogram_tv=float(.5*(np.abs(hx-hr).sum()+abs((1-hx.sum())-(1-hr.sum())))),
        outside_plot_fraction=float(1-hx.sum()))
    return metrics,{k:v for k,v in s.items() if k!='logp'}


def plot_snapshot(out,step,samples,ref,locs,stats,teacher,method):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,x,title,w in [(axes[0,0],ref,'Exact GMM40: independent reference samples',None),
                         (axes[0,1],samples,f'{method}: raw actor samples',None),
                         (axes[0,2],teacher['b'],'Importance-weighted teacher',teacher['w'])]:
        ax.hist2d(x[:,0],x[:,1],bins=160,range=[[-50,50],[-50,50]],weights=w,density=True,norm=LogNorm(vmin=1e-6,vmax=.005),cmap='magma')
        ax.scatter(locs[:,0],locs[:,1],s=7,facecolors='none',edgecolors='cyan');ax.set(title=title,xlabel='x1',ylabel='x2',aspect='equal')
    ax=axes[1,0];ax.bar(np.arange(40),stats['mass'],label='Actor')
    teacher_stats=component_stats(teacher['b'],locs,np.full_like(locs,np.log1p(np.e)),weights=teacher['w'])
    ax.plot(np.arange(40),teacher_stats['mass'],'.',color='orange',label='Teacher');ax.axhline(1/40,ls='--',color='black',label='Target 1/40')
    ax.set(title='Mode mass (posterior responsibilities)',xlabel='Target component index');ax.legend(fontsize=8)
    # Fixed component indices, not cherry-picked by quality. Include all 40 in data.
    for ax,k in zip(axes[1,1:],[0,13]):
        center=locs[k];mask=np.all(abs(samples-center)<4,axis=1)
        ax.hist2d(samples[mask,0]-center[0],samples[mask,1]-center[1],bins=50,range=[[-4,4],[-4,4]],cmap='magma')
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0,0),np.log1p(np.e),fill=False,color='cyan'))
        ax.set(title=f'Fixed target component {k}: within-mode samples',xlabel='x1 - center',ylabel='x2 - center',aspect='equal')
    fig.suptitle(f'GMM40 | step {step:,} | {len(samples):,} independent actor samples | no smoothing')
    fig.savefig(out/f'density_{step:06d}.png',dpi=150);plt.close(fig)
    A=teacher['A'];pos=teacher['pos'];b=teacher['b']
    if method.startswith('v5') or method=='gmm':pos=pos*50
    # Downsample display only by contiguous block sums; retain full matrix in npz.
    def small(a):
        rows=np.array_split(np.arange(a.shape[0]),min(256,a.shape[0]));cols=np.array_split(np.arange(a.shape[1]),min(256,a.shape[1]))
        return np.stack([np.array([a[r[:,None],c].sum() for c in cols]) for r in rows])
    # Order by nearest target mode, then x coordinate; no canonical 2D action order.
    ri=np.lexsort((pos[:,0],((pos[:,None]-locs[None])**2).sum(-1).argmin(1)))
    ci=np.lexsort((b[:,0],((b[:,None]-locs[None])**2).sum(-1).argmin(1)))
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,a,title in [(axs[0],A,'Original row/column order'),(axs[1],A[ri][:,ci],'Sorted by nearest mode, then x1')]:
        ax.imshow(small(a),aspect='auto',origin='lower',norm=LogNorm(vmin=1e-8,vmax=max(float(a.max()),1e-7)),cmap='viridis');ax.set(title=title,xlabel='Teacher (display block)',ylabel='Student (display block)')
    fig.suptitle(('GMM effective assignment w_j * responsibility' if method=='gmm' else 'Monge hard assignment / N' if method=='monge' else 'Row-normalized Sinkhorn R / N')+f' | step {step}')
    fig.savefig(out/f'assignment_{step:06d}.png',dpi=150);plt.close(fig)
