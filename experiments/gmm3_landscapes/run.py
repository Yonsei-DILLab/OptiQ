"""Population loss sections and unconstrained full-3D GD basin maps."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from ..gmm3_mean_local_minima.core import quadrature,loss

STOP=False
def stop(*args):
    global STOP
    STOP=True

def config():return json.loads(Path(__file__).with_name('config.json').read_text())

def tasks(cfg):
    return [('surface',c['id']) for c in cfg['cases']]+[('basin',k) for k in cfg['basin_cases']]+[('loggap',cfg['loggap_case'])]

def write(p,data):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def functions(case,cfg,n=None):
    x,w=quadrature(case['centers'],cfg['sigma'],n or cfg['quadrature_points']);x=jnp.asarray(x);w=jnp.asarray(w)
    f=lambda mu:loss(mu,x,w,cfg['sigma'])
    return f,jax.jit(jax.vmap(f)),jax.jit(jax.vmap(jax.value_and_grad(f))),jax.jit(jax.vmap(jax.hessian(f)))

def evaluate(fn,points,batch=128):
    chunks=[]
    for i in range(0,len(points),batch):
        p=np.asarray(points[i:i+batch]);n=len(p)
        if n<batch:p=np.pad(p,((0,batch-n),(0,0)),mode='edge')
        chunks.append(np.asarray(fn(jnp.asarray(p)))[:n])
    return np.concatenate(chunks)

def surface(case,cfg,out):
    f,vf,_,_=functions(case,cfg);c=np.asarray(case['centers']);b=np.asarray(case['anchor']);truth=float(f(jnp.asarray(c)))
    axis=np.linspace(c.min()-.5,c.max()+.5,cfg['surface_n']);xx,yy=np.meshgrid(axis,axis)
    s3=np.unique(np.r_[np.linspace(axis[0],axis[-1],cfg['third_mean_slices']),c[2],b[2]])
    cube=[]
    for z in s3:
        points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,z)]
        cube.append((evaluate(vf,points,cfg['evaluation_batch'])-truth).reshape(xx.shape))
    direction=c-b
    transverse=np.array([0.,-1.,1.]);transverse-=direction*np.dot(transverse,direction)/np.dot(direction,direction);transverse/=np.linalg.norm(transverse)
    def plane(u,v):
        uu,vv=np.meshgrid(u,v);pts=b+uu.ravel()[:,None]*direction+vv.ravel()[:,None]*transverse
        return (evaluate(vf,pts,cfg['evaluation_batch'])-truth).reshape(uu.shape)
    u=np.linspace(-.15,1.15,cfg['plane_n_u']);v=np.linspace(-1,1,cfg['plane_n_v']);z=plane(u,v)
    zu=np.linspace(-.10,.20,cfg['zoom_n']);zv=np.linspace(-.30,.30,cfg['zoom_n']);zz=plane(zu,zv)
    t=np.linspace(0,1,2001);path=b+t[:,None]*direction;path_kl=evaluate(vf,path,cfg['evaluation_batch'])-truth
    bkl=float(f(jnp.asarray(b)))-truth
    np.savez_compressed(out/'surface.npz',axis=axis,third_mean=s3,cube=np.array(cube),u=u,v=v,plane_kl=z,zoom_u=zu,zoom_v=zv,zoom_kl=zz,path_t=t,path_kl=path_kl,anchor=b,anchor_kl=bkl,truth=c,transverse=transverse,direction=direction)
    return dict(anchor_kl=bkl,straight_line_barrier=float(path_kl.max()-bkl),straight_line_peak_t=float(t[path_kl.argmax()]),note='straight-line barrier, not minimum escape barrier',third_mean_slices=s3.tolist())

def starts(case,cfg,kind):
    c=np.asarray(case['centers'])
    if kind=='basin':
        axis=np.linspace(c.min()-.5,c[2],cfg['basin_n']);ii,jj=np.triu_indices(len(axis));mu=np.c_[axis[ii],axis[jj],np.full(len(ii),c[2])]
        return mu,dict(axis=axis,ii=ii,jj=jj)
    x=np.linspace(5,5.75,cfg['loggap_n_mean']);y=np.linspace(-12,np.log10(.24),cfg['loggap_n_gap']);xx,yy=np.meshgrid(x,y)
    mu=np.c_[xx.ravel(),xx.ravel()+10**yy.ravel(),np.full(xx.size,6.)]
    return mu,dict(low_mean=x,log10_gap=y)

def basin(case,cfg,out,kind):
    f,vf,vg,hf=functions(case,cfg);initial,coords=starts(case,cfg,kind);mu=initial.copy();n=len(mu);steps=np.zeros(n,int);active=np.ones(n,bool);iteration=0
    truth=float(f(jnp.asarray(case['centers'])));times=[];history=[];cp=out/'checkpoint.npz'
    if cp.exists():
        old=np.load(cp);mu=old['mu'];steps=old['steps'];active=old['active'];iteration=int(old['iteration'])
    def one(x,_):return x-cfg['learning_rate']*vg(x)[1],None
    advance=jax.jit(lambda x:jax.lax.scan(one,x,None,length=cfg['block'])[0])
    def save():
        with (out/'checkpoint.tmp').open('wb') as stream:np.savez_compressed(stream,initial=initial,mu=mu,steps=steps,active=active,iteration=iteration)
        (out/'checkpoint.tmp').replace(cp)
    while active.any() and iteration<cfg['max_steps'] and not STOP:
        times.append(iteration);history.append(mu.copy());ids=np.flatnonzero(active);cap=max(16,1<<(len(ids)-1).bit_length())
        pack=np.pad(mu[ids],((0,cap-len(ids)),(0,0)),mode='edge');updated=advance(jnp.asarray(pack));jax.block_until_ready(updated)
        mu[ids]=np.asarray(updated)[:len(ids)];assert np.isfinite(mu).all();iteration+=cfg['block'];steps[ids]+=cfg['block']
        val,g=vg(jnp.asarray(mu));norm=np.linalg.norm(np.asarray(g),axis=-1);active=norm>cfg['gradient_tolerance']
        write(out/'STATUS.json',dict(iteration=iteration,total=n,active=int(active.sum()),stationary=int((~active).sum())));save();print(kind,case['id'],iteration,int(active.sum()),flush=True)
    if STOP:return None
    times.append(iteration);history.append(mu.copy());val,g=vg(jnp.asarray(mu));eig=np.linalg.eigvalsh(np.asarray(hf(jnp.asarray(mu))));gn=np.linalg.norm(np.asarray(g),axis=-1);kl=np.asarray(val)-truth
    # Recheck endpoint NLL/gradient/Hessian on a doubled quadrature grid.
    ff,_,vgg,hh=functions(case,cfg,cfg['validation_quadrature_points']);vv,gg=vgg(jnp.asarray(mu));he=np.linalg.eigvalsh(np.asarray(hh(jnp.asarray(mu))))
    refine=dict(loss_max_difference=float(np.max(np.abs(vv-val))),gradient_max_difference=float(np.max(np.abs(gg-g))),eigenvalue_max_difference=float(np.max(np.abs(he-eig))))
    assert refine['loss_max_difference']<1e-7 and refine['gradient_max_difference']<1e-6
    # 0 global, 1 stationary strict bad minimum, 2 stationary saddle, 3 unresolved.
    label=np.full(n,3,int);label[(gn<=cfg['gradient_tolerance'])&(eig[:,0]>1e-6)&(kl>1e-5)]=1;label[(gn<=cfg['gradient_tolerance'])&(eig[:,0]<-1e-5)]=2;label[kl<1e-6]=0
    np.savez_compressed(out/'basin.npz',initial=initial,final=mu,steps=steps,kl=kl,gradient_norm=gn,eigenvalues=eig,label=label,history=np.array(history),history_steps=np.array(times),**coords)
    return dict(counts={str(i):int((label==i).sum()) for i in range(4)},total=n,refinement=refine,max_steps=int(steps.max()),classification='0 global / 1 strict bad minimum / 2 saddle / 3 unresolved; no parameter projection')

def validate(cfg,out):
    errors=[];rng=np.random.default_rng(20260925)
    for k in ['old_symmetric_control','R1.5_D6','R1_D8']:
        case=next(c for c in cfg['cases'] if c['id']==k);c=case['centers'];pts=rng.uniform(min(c)-.5,max(c)+.5,(24,3));pts=np.r_[pts,[c],[case['anchor']]]
        _,_,vg,hf=functions(case,cfg);_,_,vg2,hf2=functions(case,cfg,cfg['validation_quadrature_points']);v,g=vg(jnp.array(pts));v2,g2=vg2(jnp.array(pts));h=hf(jnp.array(pts));h2=hf2(jnp.array(pts))
        e=dict(case=k,loss=float(np.max(np.abs(v-v2))),gradient=float(np.max(np.abs(g-g2))),hessian=float(np.max(np.abs(h-h2))));assert e['loss']<1e-7 and e['gradient']<1e-6 and e['hessian']<1e-5;errors.append(e)
    write(out,dict(passed=True,checks=errors,devices=str(jax.devices())));print(errors)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--validate',action='store_true');a=p.parse_args();cfg=config()
    if a.validate:validate(cfg,a.root/'VALIDATION.json');return
    assert json.loads((a.root/'VALIDATION.json').read_text())['passed']
    kind,cid=tasks(cfg)[a.index];case=next(c for c in cfg['cases'] if c['id']==cid);out=a.root/kind/cid;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    commit=json.loads((a.root/'SOURCE_MANIFEST.json').read_text())['commit'];write(out/'CONFIG.json',dict(config=cfg,case=case,kind=kind,commit=commit,job=os.environ.get('SLURM_JOB_ID')))
    for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    start=time.time();result=surface(case,cfg,out) if kind=='surface' else basin(case,cfg,out,kind)
    if result is not None:write(out/'COMPLETE.json',dict(case=cid,kind=kind,commit=commit,job=os.environ.get('SLURM_JOB_ID'),elapsed=time.time()-start,**result))
if __name__=='__main__':main()
