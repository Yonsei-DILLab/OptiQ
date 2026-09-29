"""Paired initialization audit using the unchanged upstream batch-32 engine."""
import argparse, json, sys, time, hashlib, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--steps', type=int, default=20000)
a=p.parse_args()
root=Path(a.source).resolve()
sys.path.insert(0,str(root))
from experiments.gmm_mode_gradient_batch32.core import *
from experiments.gmm_mode_gradient_batch32.run import verify
from optiq_dime.policy import SemiImplicitActor
from flax import serialization

manifest=verify()
assert jax.default_backend()=='gpu' and len(jax.devices())==1
assert jax.config.jax_default_matmul_precision is None
assert json.loads((root/'runtime/BASELINE_VALIDATION.json').read_text())['passed']
audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
evalz=jax.random.normal(jax.random.PRNGKey(77000+a.seed),(2048,1))
edges=np.linspace(-1,1,257)
target=np.diff(problems.analytic_cdf(edges,np.asarray(QARG)))
stepper=engine(64,64,'baseline')['block']

for variant,scale in [('baseline',1e-4),('init1',1.)]:
    folder=Path(a.out)/'runs'/f'{variant}_N64_M64_s{a.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    state,key=initialize(a.seed)
    if variant=='init1':
        model=SemiImplicitActor(1,(256,256),-5.,1.,float(np.log(.5)),mean_output_init_scale=scale)
        params=model.init(jax.random.PRNGKey(a.seed),jnp.zeros((1,1)),jnp.zeros((1,1)))['params']
        state=state.replace(params=params,opt_state=state.tx.init(params))
    config=dict(n=64,m=64,batch=32,seed=a.seed,variant=variant,mean_output_init_scale=scale,
        initial_sigma=.5,temperature=.25,q='0.25 log target density',latent='fresh normal',
        requested_steps=a.steps,source_commit=manifest['commit'],audit_commit=audit_commit,
        source_code_id=manifest['source_code_id'],devices=[str(x) for x in jax.devices()],
        engine='unchanged experiments.gmm_mode_gradient_batch32.core.engine baseline',
        omitted='counterfactual diagnostic branches only; same train engine and evaluation RNG')
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    ck=folder/'checkpoint.msgpack'
    if ck.exists():
        old=serialization.msgpack_restore(ck.read_bytes())
        assert old['source_commit']==manifest['commit']
        state=serialization.from_state_dict(state,old['actor'])
        key=jnp.asarray(old['key'])
    start=time.monotonic()
    milestones=sorted(set([0,100,500,1000,2000,5000,10000,15000,20000,a.steps]))
    for end in milestones:
        if end> a.steps or end<int(state.step):continue
        if end>int(state.step):
            state,key,avg,last=stepper(state,key,end-int(state.step))
            jax.block_until_ready(state.params)
            assert np.isfinite(np.asarray(avg)).all()
        mu,ls=heads(state.params,evalz)
        samples=np.asarray(draw(state.params,jax.random.PRNGKey(88000+a.seed)))
        hist=np.histogram(samples,edges)[0]/len(samples)
        probs=np.asarray(basin_prob(mu,ls))
        np.savez_compressed(folder/f'eval_{end:06d}.npz',samples=samples,histogram=hist,
            edges=edges,target_bin_mass=target,z=np.asarray(evalz),mu=np.asarray(mu),
            log_sigma=np.asarray(ls),basin_prob=probs)
        row=dict(seed=a.seed,variant=variant,step=end,tv=float(.5*abs(hist-target).sum()),
            basin_mass=(np.histogram(samples,[-1,-.3,.3,1])[0]/len(samples)).tolist(),
            mu_std=float(mu.std()),sigma_mean=float(jnp.exp(ls).mean()),seconds=time.monotonic()-start)
        with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        ck.write_bytes(serialization.msgpack_serialize(dict(actor=serialization.to_state_dict(state),
            key=np.asarray(key),source_commit=manifest['commit'],audit_commit=audit_commit)))
        print(json.dumps(row),flush=True)
    (folder/'COMPLETE.json').write_text(json.dumps(dict(step=int(state.step),**config),indent=2))
