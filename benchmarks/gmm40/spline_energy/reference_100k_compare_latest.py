"""Paired GMM40 comparison using the actual current private Direct implementation."""
import argparse, hashlib, importlib.metadata, json, math, os, platform, subprocess, sys, time
from pathlib import Path

UPSTREAM_SHA = "811e0e2e59a0c9137f38433e4fc18adaf5e6a8fa"
CAMP = Path("/scratch2/gsmin2024/research/optiq_latest_direct_comparison_20260921")
UPSTREAM = CAMP / "upstream"
os.environ["GMM40_REPO_ROOT"] = str(UPSTREAM)
os.environ["GMM40_RESULTS_ROOT"] = str(CAMP / "target_data")
sys.path.insert(0, str(UPSTREAM))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from scipy.special import ndtri
from scipy.stats import qmc
import wandb
from gmm40.target import initialize_target, Target
from gmm40.optiq_trg import OptiQTRG
from gmm40.evaluation import metrics as upstream_metrics
from spline_energy import (initialize_circuit, circuit_q, circuit_logp,
                           circuit_partition, circuit_sample)
from upstream.box_gaussian import mixture_log_prob


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def normalized_target(target, a):
    return target.jax_log_prob(40*a) + 2*math.log(40) - target.log_z


def spline_update(target, count=16384):
    @jax.jit
    def update(carry, _):
        state, key = carry
        key, sk, uk, mk = jax.random.split(key, 4)
        samples = circuit_sample(state.params, sk, count, parallel_root=True)
        uniform = jax.random.uniform(uk, (count,2), minval=-1., maxval=1.)
        actions = jax.lax.stop_gradient(jnp.where(
            (jax.random.uniform(mk, (count,)) < .5)[:,None], uniform, samples))
        logb = jnp.logaddexp(circuit_logp(state.params, actions)-math.log(2), -math.log(8))
        weights = jax.lax.stop_gradient(jnp.exp(normalized_target(target, actions)-logb))
        def loss(params):
            return jnp.exp(circuit_partition(params)[0])-jnp.mean(weights*circuit_q(params, actions))
        value, grad = jax.value_and_grad(loss)(state.params)
        info = dict(loss=value, grad_norm=optax.global_norm(grad),
                    target_integral_estimate=weights.mean())
        return (state.apply_gradients(grads=grad), key), info
    return update


def make_advance(step_fn, count):
    @jax.jit
    def advance(state, key):
        (state, key), info = jax.lax.scan(step_fn, (state,key), None, length=count)
        return state,key,jax.tree_util.tree_map(jnp.mean, info)
    return advance


def finite(tree):
    return all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree))


def source_record():
    root = Path(__file__).resolve().parent
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
    actual = subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()
    assert actual == UPSTREAM_SHA
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=UPSTREAM,text=True).strip()
    files = ["gmm40/optiq_trg.py", "gmm40/target.py", "gmm40/evaluation.py",
             "analysis_tools/experiments/20260920_truncated_mll/optiq_dime/policy.py",
             "analysis_tools/experiments/20260920_truncated_mll/optiq_dime/semi_implicit.py",
             "analysis_tools/experiments/20260920_truncated_mll/optiq_dime/distillation.py",
             "analysis_tools/experiments/20260920_truncated_mll/optiq_dime/box_gaussian.py"]
    return dict(source_commit=sha, upstream_commit=actual,
        upstream_url="https://github.com/Yonsei-DILLab/OptiQ/tree/"+actual,
        upstream_sha256={f:hashlib.sha256((UPSTREAM/f).read_bytes()).hexdigest() for f in files},
        dikl_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM/"gmm40-baseline/DiKL",text=True).strip(),
        command=sys.argv, hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ["jax","jaxlib","flax","optax","numpy","scipy","torch"]})


def kl_supplement(method, state, direct, target, reference, full_samples):
    tx, sx = jnp.asarray(reference/40), jnp.asarray(full_samples/40)
    lp_t = target.log_prob(reference, bounded=True) + 2*math.log(40)
    lp_s = target.log_prob(full_samples, bounded=True) + 2*math.log(40)
    if method == "spline_energy":
        logfn = jax.jit(circuit_logp)
        lt, ls = np.asarray(logfn(state.params,tx)), np.asarray(logfn(state.params,sx))
        qerr = np.asarray(circuit_q(state.params,tx)-normalized_target(target,tx))
        return dict(forward_kl=float(np.mean(lp_t-lt)), reverse_kl=float(np.mean(ls-lp_s)),
                    density="analytic native circuit",target_energy_rmse=float(np.sqrt(np.mean(qerr*qerr))),
                    log_partition=float(circuit_partition(state.params)[0]))
    # Continuous latent integral, evaluated independently from policy sampling.
    z = jnp.asarray(ndtri(qmc.Sobol(2,scramble=True,seed=8821).random_base2(16)),dtype=jnp.float32)
    mu, std = direct.actor.apply({'params':state.params},jnp.zeros((len(z),1)),z)
    @jax.jit
    def logchunk(x, m, l):
        return mixture_log_prob(x[None],m[None],l[None])[0]
    estimates = {}
    x = jnp.concatenate([tx,sx])
    for count in [32768,65536]:
        parts=[]
        for start in range(0,len(x),128):
            parts.append(np.asarray(logchunk(x[start:start+128],mu[:count],std[:count])))
        values=np.concatenate(parts)
        estimates[str(count)] = dict(forward_kl=float(np.mean(lp_t-values[:len(tx)])),
            reverse_kl=float(np.mean(values[len(tx):]-lp_s)))
    best=estimates['65536']
    return dict(**best,density="continuous-latent full policy; scrambled Sobol quadrature",
        latent_points=65536, quadrature=estimates,
        quadrature_change={k:best[k]-estimates['32768'][k] for k in best},
        caveat="KL is for full stochastic policy, not mu-only; finite quadrature and MC expectations")


def benchmark(direct, spline_state):
    @jax.jit
    def mu_fn(params,key):
        z=jax.random.normal(key,(1,2))
        return 40*direct.actor.apply({'params':params},jnp.zeros((1,1)),z)[0]
    full_fn=jax.jit(lambda p,k:direct._sample(p,k,1)[0])
    circuit_fn=jax.jit(lambda p,k:40*circuit_sample(p,k,1,parallel_root=True))
    fns=[('direct_mu_only',mu_fn,direct.state.params),('direct_full_policy',full_fn,direct.state.params),
         ('spline_native',circuit_fn,spline_state.params)]
    result={}
    keys=jax.random.split(jax.random.PRNGKey(71223),330)
    for name,fn,params in fns:
        for key in keys[:30]: jax.block_until_ready(fn(params,key))
        times=[]
        for key in keys[30:]:
            begin=time.perf_counter();jax.block_until_ready(fn(params,key))
            times.append(1000*(time.perf_counter()-begin))
        result[name]=dict(p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),calls=300)
    result['scope']='batch1, same GPU; host dispatch/sync included; no state encoder in toy circuit'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume-20k',action='store_true')
    parser.add_argument('--narrow-direct',action='store_true',help='Requested cap -3.5, initial -4; Direct starts fresh')
    parser.add_argument('--mu90-direct',action='store_true',help='Entire committed successful MU90 replication profile')
    args=parser.parse_args()
    if args.mu90_direct:args.narrow_direct=True
    assert not args.narrow_direct or args.resume_20k
    if jax.default_backend()!='gpu': raise RuntimeError('Use allocated GPU')
    source=source_record()
    target=Target()  # Target definition is prepared once before array submission.
    reference=target.sample(10000,20260921,bounded=True)
    full_reference=target.sample(10000,20260921,bounded=False)
    assert args.steps%5000==0
    start_step=20000 if args.resume_20k else 0
    assert args.steps>start_step
    suffix='mu90_100k_a1' if args.mu90_direct else ('capm35_initm4_100k_a1' if args.narrow_direct else ('100k_a1' if args.resume_20k else 'a1'))
    root=CAMP/'outputs'/f'pair_seed{args.seed}_{suffix}'
    root.mkdir(parents=True,exist_ok=False)
    cap,initial=(-3.5,-4.) if args.narrow_direct else (-1.,-1.)
    hidden_dims=(256,256);mean_init=1.;teacher_floor=math.exp(-5)
    if args.mu90_direct:
        profile_path=UPSTREAM/'gmm40/mu90_replication_plan.json'
        profile=json.loads(profile_path.read_text())
        assert profile['steps']==args.steps==100000 and profile['optiq_n']==profile['optiq_m']==64
        assert profile['batch']==256 and profile['temperature']==1
        actor_cfg=profile['trg_actor']
        cap,initial=actor_cfg['log_std_max'],actor_cfg['initial_log_std']
        assert (cap,initial)==(-3.5,-4.)
        hidden_dims=(profile['width'],)*profile['depth']
        mean_init=actor_cfg['mean_output_init_scale'];teacher_floor=actor_cfg['teacher_std_floor']
        source['direct_profile']=dict(path=str(profile_path),sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),profile=profile,
            seeds_note='Use seeds0..4; source plan adds1..3 to its separately retained seed0. Seed4 is additional validation.')
    direct=OptiQTRG(target,args.seed,log_std_max=cap,initial_log_std=initial,
                    hidden_dims=hidden_dims,mean_output_init_scale=mean_init,teacher_std_floor=teacher_floor)
    state,key=initialize_circuit(args.seed)
    cstep=spline_update(target)
    previous={}
    if args.resume_20k:
        assert args.steps==100000
        parent=CAMP/'outputs'/f'pair_seed{args.seed}_a1'
        assert (parent/'COMPLETE.json').exists()
        for method,st,k in [('direct_gmm_trg',direct.state,direct.key),('spline_energy',state,key)]:
            if args.narrow_direct and method.startswith('direct'):continue
            checkpoint=parent/method/'checkpoint_020000.msgpack'
            restored=serialization.from_bytes(dict(state=st,key=k),checkpoint.read_bytes())
            assert int(restored['state'].step)==20000 and finite(restored)
            roundtrip=serialization.from_bytes(restored,serialization.to_bytes(restored))
            assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
                jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves(roundtrip)))
            previous[method]=dict(manifest=json.loads((parent/method/'manifest.json').read_text()),
                latest=json.loads((parent/method/'latest.json').read_text()),
                checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            if method.startswith('direct'):direct.state,direct.key=restored['state'],restored['key']
            else:state,key=restored['state'],restored['key']
        source['resume_from']=previous
        source['training_origin_commits']={m:v['manifest']['source_commit'] for m,v in previous.items()}
        source['fresh_direct_requested_sigma']=args.narrow_direct
    # New coordinate/gradient check plus native-batch timing; discard gate states.
    np.testing.assert_allclose(np.asarray(normalized_target(target,jnp.asarray(reference[:32]/40))),
        target.log_prob(reference[:32],bounded=True)+2*math.log(40),rtol=2e-5,atol=2e-5)
    gate={}
    gates=([('direct',direct._update,direct.state,direct.key)] if args.narrow_direct else []) if args.resume_20k else [('direct',direct._update,direct.state,direct.key),('spline',cstep,state,key)]
    for name,step,st,k in gates:
        fn=make_advance(step,2)
        begin=time.monotonic();compiled=fn.lower(st,k).compile();comp=time.monotonic()-begin
        begin=time.monotonic();ns,nk,info=compiled(st,k);jax.block_until_ready(ns)
        assert int(ns.step)==2 and finite((ns.params,info))
        delta=max(float(jnp.max(abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(ns.params),jax.tree_util.tree_leaves(st.params)))
        assert delta>0
        gate[name]=dict(compile_seconds=comp,two_update_seconds=time.monotonic()-begin,max_parameter_change=delta,
                       info={k:float(v) for k,v in info.items()})
    if args.resume_20k:
        gate['restored_models']=dict(optimizer_step=20000,parameters_optimizer_rng_roundtrip_exact=True,
                  no_hyperparameter_change=True,checkpoint_hashes={m:v['checkpoint_sha256'] for m,v in previous.items()})
    write_json(root/'preflight.json',dict(passed=True,checks=gate,source=source))
    print(json.dumps(dict(preflight=gate)),flush=True)
    if args.check_only: return
    final_states={}
    for method in ['direct_gmm_trg','spline_energy']:
        out=root/method
        # A completed circuit is independent of which Direct profile shared its job.
        reusable=CAMP/'outputs'/f'pair_seed{args.seed}_capm35_initm4_100k_a1'/'spline_energy'
        if args.mu90_direct and method=='spline_energy' and (reusable/'COMPLETE.json').exists():
            origin=json.loads((reusable/'manifest.json').read_text())
            expected=dict(method='spline_energy',steps=100000,seed=args.seed,rank=64,knots=129,
                lr=3e-4,defensive_uniform=.5,target_queries_per_update=16384,scale=40,temperature=1)
            assert all(origin['config'][k]==v for k,v in expected.items())
            checkpoint=reusable/'checkpoint_100000.msgpack'
            restored=serialization.from_bytes(dict(state=state,key=key),checkpoint.read_bytes())
            assert int(restored['state'].step)==100000 and finite(restored)
            out.symlink_to(reusable,target_is_directory=True)
            final_states[method]=restored['state']
            write_json(root/'reused_spline.json',dict(source=str(reusable),manifest=origin,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),no_retraining=True))
            alias=wandb.init(entity='OptiQ',project='OptiQ-GMM40-Sampling-Comparison',
                group='latest-direct-811e0e2-mu90-100k',job_type='reused_checkpoint',
                name=f'GMM40-MU90-comparison-spline-seed{args.seed}-REUSED-100k',
                id=hashlib.sha256(str(out).encode()).hexdigest()[:12],resume='never',dir=str(root),
                config=dict(origin['config'],reused=True,reused_from=str(reusable),source_commit=origin['source_commit']))
            alias.define_metric('update');alias.define_metric('*',step_metric='update')
            rows=[json.loads(line) for line in (reusable/'history.jsonl').read_text().splitlines()]
            parent=origin.get('resume_from',{}).get('spline_energy')
            if parent:
                older=[json.loads(line) for line in (Path(parent['checkpoint']).parent/'history.jsonl').read_text().splitlines()]
                rows=[row for row in older if row['update']<rows[0]['update']]+rows
            for row in rows:
                payload={k:row[k] for k in ['update','training_seconds','target_queries']}
                for view in ['primary','full_policy']:
                    payload.update({f'{view}/{n}':v for n,v in row[view].items() if isinstance(v,(int,float))})
                alias.log(payload,step=row['update'])
            alias.summary['reused_checkpoint']=str(checkpoint)
            alias.summary['full_density_kl']=json.loads((reusable/'kl.json').read_text())
            alias.summary.update(payload)
            alias_url=alias.url;alias.finish()
            write_json(root/'reused_wandb.json',dict(url=alias_url,reused=True,no_new_training=True))
            continue
        out.mkdir()
        algorithm='OptiQ Direct GMM/TRG (MU90 40-mode profile)' if args.mu90_direct else ('OptiQ Direct GMM/TRG (cap -3.5, init -4)' if args.narrow_direct else 'OptiQ Direct GMM/TRG (latest default)')
        cfg=dict(method=method,algorithm=algorithm,
            seed=args.seed,steps=args.steps,target_queries_per_update=16384,target_queries=args.steps*16384,
            target='DiKL GMM40 seed0 conditioned on (-40,40)^2',scale=40,temperature=1,lr=3e-4,
            eval_samples=10000,eval_every=5000,upstream_commit=UPSTREAM_SHA,
            n=64,m=64,batch=256,hidden_dims=list(hidden_dims),log_std_min=-5,log_std_max=cap,
            initial_log_std=initial,mean_init_scale=mean_init,teacher_std_floor=teacher_floor,latent_prior='continuous_normal') if method.startswith('direct') else dict(
            method=method,algorithm='Spline Energy Circuit',seed=args.seed,steps=args.steps,target_queries_per_update=16384,
            target_queries=args.steps*16384,target='DiKL GMM40 seed0 conditioned on (-40,40)^2',scale=40,
            temperature=1,lr=3e-4,eval_samples=10000,eval_every=5000,rank=64,knots=129,defensive_uniform=.5,
            state_conditioner=False,loss='Poisson energy fitting',initialization='target-independent 8x8 covering grid')
        if method.startswith('direct'):
            st,k=direct.state,direct.key;step=direct._update
        else: st,k=state,key;step=cstep
        cfg['trainable_parameters']=sum(v.size for v in jax.tree_util.tree_leaves(st.params))
        if method in previous:
            original=previous[method]['manifest']['config']
            for field,value in cfg.items():
                if field not in ['steps','target_queries']:assert original[field]==value,(field,original[field],value)
        write_json(out/'manifest.json',dict(config=cfg,**source,output=str(out)))
        group='latest-direct-811e0e2'+('-mu90-100k' if args.mu90_direct else ('-capm35-initm4-100k' if args.narrow_direct else ('-100k' if args.resume_20k else '')))
        run=wandb.init(entity='OptiQ',project='OptiQ-GMM40-Sampling-Comparison',group=group,
            name=f'GMM40-{suffix}-{method}-seed{args.seed}',id=hashlib.sha256(str(out).encode()).hexdigest()[:12],
            resume='never',dir=str(out),config=dict(cfg,source_commit=source['source_commit']),job_type='train',
            settings=wandb.Settings(init_timeout=120))
        write_json(out/'wandb.json',dict(id=run.id,url=run.url))
        run.define_metric('update');run.define_metric('*',step_metric='update')
        advance=make_advance(step,5000)
        begin=time.monotonic();advance=advance.lower(st,k).compile();comp=time.monotonic()-begin
        run.summary['compile_seconds']=comp
        method_start_step=20000 if method in previous else 0
        training_seconds=previous[method]['latest']['training_seconds'] if method in previous else 0.;info={}
        with (out/'history.jsonl').open('w') as history:
            for iteration in range(method_start_step,args.steps+1,5000):
                if iteration>method_start_step:
                    begin=time.monotonic();st,k,info=advance(st,k);jax.block_until_ready(st)
                    training_seconds+=time.monotonic()-begin
                assert int(st.step)==iteration and finite((st.params,info))
                evalkey=jax.random.PRNGKey(9102026)
                if method.startswith('direct'):
                    full,mu,_,_=direct.sample_fn(st.params,evalkey,10000)
                    samples,full=np.asarray(mu),np.asarray(full)
                else:
                    samples=np.asarray(jax.jit(lambda p,key:40*circuit_sample(p,key,10000,parallel_root=True))(st.params,evalkey))
                    full=samples
                primary=upstream_metrics(samples,target,reference,full_reference)
                supplement=upstream_metrics(full,target,reference,full_reference) if method.startswith('direct') else primary
                row=dict(update=iteration,optimizer_step=int(st.step),target_queries=iteration*16384,
                    training_seconds=training_seconds,primary_view='mu_only' if method.startswith('direct') else 'native',
                    primary=primary,full_policy=supplement,train={k:float(v) for k,v in info.items()})
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush();write_json(out/'latest.json',row)
                np.savez_compressed(out/f'samples_{iteration:06d}.npz',primary=samples,full_policy=full,reference=reference)
                (out/f'checkpoint_{iteration:06d}.msgpack').write_bytes(serialization.to_bytes(dict(state=st,key=k)))
                payload=dict(update=iteration,target_queries=iteration*16384,training_seconds=training_seconds)
                for label,values in [('primary',primary),('full_policy',supplement)]:
                    payload.update({f'{label}/{n}':v for n,v in values.items() if isinstance(v,(int,float))})
                run.log(payload,step=iteration);run.summary.update(payload)
                print(json.dumps(dict(method=method,step=iteration,seconds=training_seconds,
                    near=primary['high_density_fraction'],coverage=primary['mode_coverage'])),flush=True)
        final_states[method]=st
        kl=kl_supplement(method,st,direct,target,reference,full)
        write_json(out/'kl.json',kl);run.summary['full_density_kl']=kl
        run.summary['source']=source
        artifact=wandb.Artifact(f'latest-{method}-seed{args.seed}',type='gmm40-comparison')
        for name in ['manifest.json','history.jsonl','kl.json',f'checkpoint_{args.steps:06d}.msgpack',f'samples_{args.steps:06d}.npz']:
            artifact.add_file(str(out/name))
        run.log_artifact(artifact);run.finish()
        write_json(out/'COMPLETE.json',dict(steps=int(st.step),wandb_synced=True))
        if method.startswith('direct'): direct.state=st;direct.key=k
    timing=benchmark(direct,final_states['spline_energy'])
    write_json(root/'inference_timing.json',timing)
    write_json(root/'COMPLETE.json',dict(steps=args.steps,seed=args.seed,source_commit=source['source_commit']))


if __name__=='__main__': main()
