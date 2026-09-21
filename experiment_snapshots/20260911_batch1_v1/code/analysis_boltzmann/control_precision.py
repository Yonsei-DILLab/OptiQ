"""Read-only, fixed-shape float64 quadrature audit of saved control critics."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import logsumexp
from .movecar import Critic


def evaluator(params,chunk=4096,dtype='float64'):
    net=Critic()
    @jax.jit
    def apply(p,s,a): return net.apply({'params':p},s,a)
    def q(states,actions):
        arrays=[]
        with jax.experimental.enable_x64(), jax.default_matmul_precision('highest'):
            dt=jnp.float64 if dtype=='float64' else jnp.float32
            p=jax.tree.map(lambda x:jnp.asarray(x,dtype=dt),params)
            ss=np.broadcast_to(np.asarray(states)[:,None],(len(states),len(actions))).reshape(-1,1)
            aa=np.broadcast_to(np.asarray(actions)[None,:],(len(states),len(actions))).reshape(-1,1)
            for start in range(0,len(ss),chunk):
                size=min(chunk,len(ss)-start)
                sc=np.pad(ss[start:start+size],((0,chunk-size),(0,0)),mode='edge')
                ac=np.pad(aa[start:start+size],((0,chunk-size),(0,0)),mode='edge')
                arrays.append(np.asarray(apply(p,jnp.asarray(sc,dtype=dt),jnp.asarray(ac,dtype=dt)),dtype=np.float64)[:size])
        return np.concatenate(arrays).reshape(len(states),len(actions))
    return q


def moments(values,tau=.25):
    # Center the expectation as well as logits to avoid cancellation of O(200) Q.
    mx=values.max(axis=1,keepdims=True);delta=values-mx
    w=np.exp(delta/tau-logsumexp(delta/tau,axis=1,keepdims=True))
    return mx[:,0]+np.sum(w*delta,axis=1)


def audit(run,out,states_count=256,max_n=131584,tol=1e-4):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=max(run.glob('checkpoint_*.msgpack'),key=lambda p:int(p.stem.split('_')[-1]))
    state=serialization.msgpack_restore(checkpoint.read_bytes())
    params=state['critic']['target_params']
    q=evaluator(params);q32=evaluator(params,dtype='float32')
    states=np.asarray(jnp.linspace(0,10,states_count),dtype=np.float64)
    hist=[];prev=None;consecutive=0;coarse=None;fine=None;start=time.monotonic()
    training_grid=int(json.loads((run/'manifest.json').read_text())['arguments'].get('grid',257))
    n=training_grid
    while n<=max_n:
        actions=-1+(np.arange(n,dtype=np.float64)+.5)*2/n
        vals=q(states,actions);b=moments(vals)
        if coarse is None:coarse=b.copy()
        if n==training_grid*2:fine=b.copy()
        err=None if prev is None else float(abs(b-prev).max())
        hist.append({'n':n,'max_delta':err,'value_min':float(b.min()),'value_max':float(b.max()),'elapsed':time.monotonic()-start})
        print(json.dumps({'run':run.name,**hist[-1]}),flush=True)
        consecutive=consecutive+1 if err is not None and err<tol else 0
        np.savez_compressed(out/'values.npz',states=states,coarse=coarse,reference=b)
        (out/'history.json').write_text(json.dumps(hist,indent=2))
        if consecutive>=2:break
        prev=b;n*=2
    # Isolate float32 Q evaluation from quadrature discretization on the same grid.
    actions=-1+(np.arange(training_grid,dtype=np.float64)+.5)*2/training_grid
    b32=moments(q32(states,actions))
    result={'run':run.name,'checkpoint':checkpoint.name,'states':states_count,'training_grid':training_grid,'reference_n':hist[-1]['n'],'converged':consecutive>=2,'tolerance':tol,'consecutive_refinements':2,'coarse_to_reference':float(abs(coarse-b).max()),'coarse_to_fine64':None if fine is None else float(abs(coarse-fine).max()),'fixed_shape_float32_vs64':float(abs(b32-coarse).max()),'history':hist}
    if states_count==256 and (run/'grid_convergence.npz').exists():
        old=np.load(run/'grid_convergence.npz')
        result['old_float32_refinement_error']=float(old['max_error'])
        result['old_coarse_vs_float64']=float(abs(old['coarse']-coarse).max())
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result

def install_validation(run,audit_dir):
    run=Path(run); audit_dir=Path(audit_dir)
    info=json.loads((audit_dir/'summary.json').read_text())
    arrays=np.load(audit_dir/'values.npz')
    if not info['converged']: raise RuntimeError('Float64 quadrature did not converge: '+str(audit_dir))
    np.savez_compressed(run/'grid_convergence.npz',coarse=arrays['coarse'],fine=arrays['reference'],max_error=info['coarse_to_reference'],reference_n=info['reference_n'],training_grid=info['training_grid'],reference_converged=True,reference_tolerance=info['tolerance'])
    (run/'reference_precision.json').write_text(json.dumps(info,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);ap.add_argument('--states',type=int,default=256);ap.add_argument('--max-n',type=int,default=131584);args=ap.parse_args()
    audit(args.run,args.out,args.states,args.max_n)
