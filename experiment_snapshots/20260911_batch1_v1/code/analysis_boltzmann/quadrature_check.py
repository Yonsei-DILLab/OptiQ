"""Diagnose midpoint refinement with a fixed-shape critic evaluation kernel."""
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from scipy.special import logsumexp
from .movecar import make_agent

def precise_q(qstate):
    # Promote the saved float32 weights only during diagnostic Q evaluation.
    # Actor sampling and all training computations retain their original dtype.
    @jax.jit
    def evaluate(obs,a):
        params=jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.float64),qstate.target_params)
        return qstate.apply_fn({'params':params,'batch_stats':qstate.target_batch_stats},jnp.asarray(obs,dtype=jnp.float64),jnp.asarray(a,dtype=jnp.float64),train=False)[...,0].min(0)
    def q(obs,a):
        with jax.experimental.enable_x64():
            return evaluate(obs,a)
    return q

def reference(q,state,tol=1e-4,max_n=131072):
    previous=None; history=[]; consecutive=0
    for n in (512,1024,2048,4096,8192,16384,32768,65536,131072):
        if n>max_n: break
        a=(-1+(np.arange(n,dtype=np.float64)+.5)*2/n).astype('float32')[:,None]
        chunks=[]
        for start in range(0,n,2048):
            chunk=a[start:start+2048]; size=len(chunk)
            padded=np.pad(chunk,((0,2048-size),(0,0)),mode='edge')
            chunks.append(np.asarray(q(np.full_like(padded,state),padded),dtype='float64')[:size])
        vals=np.concatenate(chunks); mass=np.exp(vals/.25-logsumexp(vals/.25)); truth=float(mass@vals)
        delta=None if previous is None else abs(truth-previous)
        history.append({'n':n,'value':truth,'delta':delta})
        consecutive=consecutive+1 if delta is not None and delta<tol else 0
        if consecutive>=2: return truth,a[vals.argmax()],n,history
        previous=truth
    raise RuntimeError(json.dumps({'state':float(state),'quadrature_history':history}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--campaign',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    rows=[]
    for seed in (0,1,3):
        agent=make_agent('optiq',seed); agent.load(Path(args.campaign)/'runs'/f'movecar_optiq_seed{seed}'/'checkpoint_1000000.msgpack'); st=agent.model.policy.qf_state
        q=precise_q(st)
        for state in np.linspace(0,10,32):
            value,center,n,history=reference(q,state)
            rows.append({'seed':seed,'state':float(state),'n':n,'history':history})
            print(json.dumps({'seed':seed,'state':float(state),'n':n,'final_delta':history[-1]['delta']}),flush=True)
        jax.clear_caches()
    Path(args.out).write_text(json.dumps(rows,indent=2))
