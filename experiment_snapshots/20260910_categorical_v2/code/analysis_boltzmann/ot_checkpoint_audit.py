"""Post-hoc inspection of existing five-seed actors; no training or tuning.
Replay production candidate/weight/coupling/row-argmax calculations and check
loss + source-ESS parity against the actual update function (discard its update).
"""
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,jax.scipy as jsp
from flax import serialization
from .shared import actor_state,config,dummy_critic,ot_update
from .problems import make_problem
from optiq_dime.transport import TruncatedGaussianKDE,sample_truncated_gaussian_mixture,sinkhorn


def build(case):
    p=make_problem(case);st=actor_state(p.dim,0);cfg=config().alg.actor
    assert cfg.transport_target_mode=='argmax' and not cfg.adaptive_density_beta
    n=int(cfg.num_policy_samples);m=int(cfg.proposals_per_policy_sample);batch=256
    @jax.jit
    def forward(params,key):
        _,lk,pk,dk=jax.random.split(key,4)
        z=jax.random.normal(lk,(batch,n,p.dim),dtype=jnp.float32)
        raw=st.apply_fn({'params':params},jnp.zeros((batch*n,1)),z.reshape(-1,p.dim)).reshape(batch,n,p.dim)
        actions=jnp.clip(raw,-1,1)
        kde=TruncatedGaussianKDE.from_centers(actions,float(cfg.proposal_std),float(cfg.proposal_clip))
        if cfg.proposal_sampling_mode=='stratified': candidates=kde.sample_stratified(pk,m,bool(cfg.include_anchor))
        else:candidates=sample_truncated_gaussian_mixture(pk,kde.centers,m,float(cfg.proposal_std),float(cfg.proposal_clip),bool(cfg.include_anchor),return_component_indices=True)[0]
        candidates=candidates.reshape(batch,n*m,p.dim);values=p.q(candidates,jnp,jsp.special.logsumexp)
        density=kde.log_prob(candidates) if cfg.density_correction else jnp.zeros_like(values)
        weights=jax.nn.softmax(values/float(cfg.temperature)-float(cfg.density_beta)*density,axis=-1)
        cost=jnp.sum((actions[:,:,None,:]-candidates[:,None,:,:])**2,axis=-1);cost=cost/(cost.mean((-2,-1),keepdims=True)+1e-8)
        coupling=sinkhorn(cost,weights,float(cfg.sinkhorn_epsilon),int(cfg.sinkhorn_iterations))
        row=coupling/jnp.maximum(coupling.sum(-1,keepdims=True),1e-20);ids=jnp.argmax(row,axis=-1)
        targets=jax.vmap(lambda a,i:a[i])(candidates,ids)
        hard=jnp.mean(jax.nn.one_hot(ids,n*m),axis=1)
        row_expected=row.mean(axis=1)
        longer=sinkhorn(cost,weights,float(cfg.sinkhorn_epsilon),300)
        long_row=longer/jnp.maximum(longer.sum(-1,keepdims=True),1e-20)
        long_ids=jnp.argmax(long_row,axis=-1)
        long_targets=jax.vmap(lambda a,i:a[i])(candidates,long_ids)
        return {'row_expected':row_expected,'long_row_expected':long_row.mean(axis=1),'long_targets':long_targets,'row_tv':jnp.mean(jnp.sum(abs(row_expected-weights),axis=1)/2),'long_row_l1':jnp.mean(jnp.sum(abs(longer.sum(2)-1/n),axis=1)),'long_row_tv':jnp.mean(jnp.sum(abs(long_row.mean(axis=1)-weights),axis=1)/2),'actions':actions,'candidates':candidates,'weights':weights,'column':coupling.sum(1),'targets':targets,'ess':jnp.mean(1/jnp.sum(weights**2,axis=1)),'loss':jnp.mean(jnp.sum((raw-targets)**2,axis=-1)),'hard_tv':jnp.mean(jnp.sum(abs(hard-weights),axis=1)/2),'row_l1':jnp.mean(jnp.sum(abs(coupling.sum(2)-1/n),axis=1)),'column_l1':jnp.mean(jnp.sum(abs(coupling.sum(1)-weights),axis=1))}
    @jax.jit
    def jacobian(params,key):
        z=jax.random.normal(key,(128,p.dim))
        def f(v):return jnp.clip(st.apply_fn({'params':params},jnp.zeros((1,1)),v[None,:])[0],-1,1)
        js=jax.vmap(jax.jacfwd(f))(z);sv=jnp.linalg.svd(js,compute_uv=False)
        return jnp.median(sv[:,-1]/jnp.maximum(sv[:,0],1e-12))
    critic=dummy_critic(lambda obs,a:p.q(a,jnp,jsp.special.logsumexp))
    return p,st,forward,jacobian,critic

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();root=Path(args.campaign)/'runs';out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];parity=[]
    for case in ['unimodal','modes2d_4','modes2d_8','separable_8']:
        p,template,forward,jac,critic=build(case)
        for seed in range(5):
            for init in ['default','coverage']:
                d=root/f'frozen_{case}_{init}_seed{seed}'
                for step in [0,20000]:
                    state=serialization.from_bytes(template,(d/f'actor_{step}.msgpack').read_bytes());ref=np.load(d/f'distribution_{step}.npz')['reference_mode_mass'];L=len(ref)
                    mass={k:np.zeros(L) for k in ['actor','candidate','weighted','transport','row_expected','long_row_expected','hard','long_hard']};vals={k:[] for k in mass};tele=[];actions=[]
                    for rep in range(4):
                        key=jax.random.PRNGKey(700000+seed*100+rep);x={k:np.asarray(v) for k,v in forward(state.params,key).items()}
                        if seed==0 and init=='default' and step==0 and rep==0:
                            _,loss,_,metrics=ot_update(state,critic,jnp.zeros((256,1)),key)
                            a=float(x['loss']);b=float(loss);c=float(x['ess']);e=float(metrics['source_ess_absolute']);assert np.isclose(a,b,rtol=2e-5,atol=2e-6),(case,'loss',a,b);assert np.isclose(c,e,rtol=2e-5,atol=2e-6),(case,'ess',c,e);parity.append({'case':case,'loss_difference':abs(a-b),'ess_difference':abs(c-e)})
                        for k,points,w in [('actor',x['actions'],None),('candidate',x['candidates'],None),('weighted',x['candidates'],x['weights']/256),('transport',x['candidates'],x['column']/256),('row_expected',x['candidates'],x['row_expected']/256),('long_row_expected',x['candidates'],x['long_row_expected']/256),('hard',x['targets'],None),('long_hard',x['long_targets'],None)]:
                            labels=p.labels(points).ravel();w=np.ones(len(labels))/len(labels) if w is None else w.ravel();bins=np.bincount(labels,weights=w,minlength=L);assert len(bins)<=L;mass[k]+=bins/4;vals[k].append(float(np.sum(p.q(points).ravel()*w)))
                        tele.append({k:float(x[k]) for k in ['ess','loss','hard_tv','row_l1','column_l1','row_tv','long_row_l1','long_row_tv']});actions.append(x['actions'].reshape(-1,p.dim))
                    row={'case':case,'seed':seed,'initialization':init,'step':step,'input_latent_dim':int(state.params['Dense_0']['kernel'].shape[0])-1,'output_dim':p.dim,'jacobian_min_max_ratio':float(jac(state.params,jax.random.PRNGKey(710000+seed))),'telemetry':{k:float(np.mean([t[k] for t in tele])) for k in tele[0]},'bin_tv':{k:float(abs(m-ref).sum()/2) for k,m in mass.items()},'q_mean':{k:float(np.mean(v)) for k,v in vals.items()},'mass':{k:v.tolist() for k,v in mass.items()},'reference_mass':ref.tolist()}
                    rows.append(row);(out/'summary.json').write_text(json.dumps({'rows':rows,'parity':parity,'repetitions':4,'groups_per_repetition':256,'training_seeds':list(range(5))},indent=2));print(json.dumps({k:row[k] for k in ['case','seed','initialization','step','telemetry','bin_tv']}),flush=True)
        jax.clear_caches()
    (out/'COMPLETE').write_text('post-hoc audit complete; no parameter updates retained\n')
