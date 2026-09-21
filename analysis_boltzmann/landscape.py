"""Critic-induced value objectives, NOT the OT regression training objective."""
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .movecar import make_agent
from .shared import begin


def run(args):
    out=begin(args.out,args)
    actor=make_agent(args.method,args.seed); actor.load(args.start)
    def state_of(agent):
        return agent.model.policy.actor_state if args.method=='optiq' else agent.actors[0]
    st=state_of(actor); initial,unravel=ravel_pytree(st.params)
    endpoint=make_agent(getattr(args,'end_method',None) or args.method,args.seed); endpoint.load(args.end)
    final,_=ravel_pytree(state_of(endpoint).params)
    obs=jnp.repeat(jnp.linspace(0,10,256)[:,None],32,axis=0)
    z=jax.random.normal(jax.random.PRNGKey(30000+args.seed),(len(obs),1))
    critics=args.critics or [f'{args.method}:{args.end}']
    rng=np.random.default_rng(args.seed+40000)
    directions=rng.normal(size=(args.directions,len(final))).astype('float32')
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    records={}; metrics={}
    for entry in critics:
        name,path=entry.split(':',1); critic=make_agent(name,args.seed); critic.load(path)
        if name=='optiq':
            qs=critic.model.policy.qf_state
            def q(s,a): return qs.apply_fn({'params':qs.params,'batch_stats':qs.batch_stats},s,a,train=False)[...,0].mean(0)
        else:
            qs=critic.critics
            def q(s,a): return qs[0].apply_fn({'params':qs[0].params},s,a)
        def objective(flat):
            params=unravel(flat)
            action=jnp.clip(st.apply_fn({'params':params},obs,z),-1,1) if args.method=='optiq' else st.apply_fn({'params':params},obs)
            return -jnp.mean(q(obs,action))
        objective=jax.jit(objective)
        alpha=np.linspace(0,1,101)
        values=np.array([float(objective((1-a)*initial+a*final)) for a in alpha])
        records[name+'_alpha']=alpha; records[name+'_path']=values
        metrics[name]={'barrier_above_endpoints':float(values.max()-max(values[0],values[-1]))}
        for checkpoint,center in [('start',initial),('end',final)]:
            base=float(objective(center)); norm=float(jnp.linalg.norm(center))
            for radius in (.01,.05,.1):
                delta=[]
                for d in directions:
                    delta.append([float(objective(center+radius*norm*d))-base,float(objective(center-radius*norm*d))-base])
                delta=np.asarray(delta); label=f'{name}_{checkpoint}_r{radius}'
                records[label]=delta
                metrics[label]={'decreasing_direction_fraction':float(np.mean(delta.min(1)<-1e-6)),'flat_direction_fraction':float(np.mean(np.max(abs(delta),axis=1)<1e-6))}
    np.savez_compressed(out/'landscape.npz',**records)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--method',default='optiq'); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--end-method'); ap.add_argument('--start',required=True); ap.add_argument('--end',required=True); ap.add_argument('--critics',nargs='*'); ap.add_argument('--directions',type=int,default=256); ap.add_argument('--out',required=True)
    run(ap.parse_args())
