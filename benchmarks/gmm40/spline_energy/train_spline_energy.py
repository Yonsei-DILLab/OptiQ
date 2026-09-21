"""Train the one-box spline energy model; preserve the legacy GMM40 evaluator."""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from core import target_parameters, target_logp, Actor
from upstream.box_gaussian import sample_box
from train import write_json, code_state, evaluation_set, labels
from spline_energy import (initialize_circuit, circuit_q, circuit_logp,
                           circuit_partition, circuit_sample, make_circuit_update)

ROOT = Path(__file__).resolve().parent
PROJECT = "OptiQ-GMM40-Sampling-Comparison"

@jax.jit
def eval_arrays(params, target, tx, ux):
    samples = circuit_sample(params, jax.random.PRNGKey(9102026), tx.shape[0])
    return (samples, circuit_logp(params, tx), circuit_logp(params, samples),
            target_logp(samples,target), circuit_q(params,ux)-target_logp(ux,target),
            circuit_partition(params)[0])

def evaluate(state, target, evaluation, uniform):
    tx, tlp, refmass = evaluation
    sample, tlq, lq, lp, errors, logz = map(np.asarray, eval_arrays(state.params,target,tx,uniform))
    assignment = labels(sample,np.asarray(target[0]))
    mass = np.bincount(assignment,minlength=40)/len(sample)
    distances = (((sample[:,None]-np.asarray(target[0])[None])/np.exp(np.asarray(target[1]))[None])**2).sum(-1)
    near = distances.min(-1) < 9.
    near_mass = np.bincount(assignment,weights=near.astype(float),minlength=40)/len(sample)
    metrics = dict(forward_kl=float(np.mean(np.asarray(tlp)-tlq)), reverse_kl=float(np.mean(lq-lp)),
        target_nll=float(-tlq.mean()), precision_3sigma=float(near.mean()),
        mode_coverage=int((mass >= .1*refmass).sum()),
        mode_coverage_3sigma=int((near_mass >= .1*refmass).sum()),
        mode_mass_tv=float(.5*abs(mass-refmass).sum()),
        min_mode_mass=float(mass.min()), max_mode_mass=float(mass.max()),
        log_partition=float(logz), partition_error=float(abs(np.exp(logz)-1.)),
        energy_rmse_uniform=float(np.sqrt(np.mean(errors**2))),
        energy_mae_uniform=float(np.mean(abs(errors))))
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite evaluation")
    return metrics,sample,mass,refmass

def plot(path, target, sample, mass, refmass, title):
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].scatter(*(sample[:6000]*50).T,s=2,alpha=.2,color="#377eb8")
    axes[0].scatter(*(np.asarray(target[0])*50).T,marker="x",color="#e66101",s=25,label="Target modes")
    axes[0].set(xlim=(-50,50),ylim=(-50,50),xlabel="x1",ylabel="x2",aspect="equal")
    axes[0].legend()
    ids=np.arange(40)
    axes[1].bar(ids-.2,refmass,.4,color="#e66101",label="Target")
    axes[1].bar(ids+.2,mass,.4,color="#377eb8",label="Spline energy policy")
    axes[1].set(xlabel="Nearest target mode",ylabel="Probability",ylim=(0,max(.06,mass.max()*1.1)))
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(path,dpi=160)
    plt.close(fig)

def inference_benchmark(state, seed, out):
    # No parameter caching/constant folding. Include circuit normalization/CDF.
    circuit=jax.jit(lambda p,k: circuit_sample(p,k,1))
    basepath=ROOT.parent.parent if ROOT.parent.name=="snapshots" else ROOT.parent
    baseline=basepath/"outputs"/f"smem_compare_snis_seed{seed}_a1"/"checkpoint_020000.msgpack"
    # Snapshots live CAMP/snapshots/name, so CAMP is parents[1].
    if not baseline.exists():
        baseline=ROOT.parents[1]/"outputs"/f"smem_compare_snis_seed{seed}_a1"/"checkpoint_020000.msgpack"
    methods=[("spline_energy",circuit,state.params)]
    if baseline.exists():
        data=serialization.msgpack_restore(baseline.read_bytes())
        p=jax.tree_util.tree_map(jnp.asarray,data["state"]["params"])
        z=jnp.asarray(data["z"])
        actor=Actor()
        @jax.jit
        def direct(p,k):
            ik,nk=jax.random.split(k)
            i=jax.random.randint(ik,(),0,len(z))
            mu,ls=actor.apply({"params":p},z[i][None])
            return sample_box(nk,mu,ls)
        methods.append(("direct_gmm",direct,p))
    result={}
    keys=jax.random.split(jax.random.PRNGKey(778899),120)
    for name,fn,p in methods:
        jax.block_until_ready(fn(p,keys[0]))
        for k in keys[:20]: jax.block_until_ready(fn(p,k))
        elapsed=[]
        for k in keys[20:]:
            start=time.perf_counter()
            jax.block_until_ready(fn(p,k))
            elapsed.append(1000*(time.perf_counter()-start))
        result[name]=dict(p50_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
            repetitions=len(elapsed),batch=1,includes_host_dispatch=True,critic_calls=0,
            denoising_steps=0)
    result["baseline_checkpoint"]=str(baseline)
    result["caveat"]="State-free toy; no trained state-conditioning MLP in circuit. Same-device batch-1 sampler timing is not MuJoCo latency."
    write_json(out/"inference_timing.json",result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--steps",type=int,default=20000)
    parser.add_argument("--eval-every",type=int,default=1000)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",default="spline_energy_v1")
    parser.add_argument("--attempt",default="a1")
    args=parser.parse_args()
    assert args.steps % args.eval_every == 0
    sha=code_state()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(method="spline_energy",algorithm="Spline Energy Circuit",seed=args.seed,
        steps=args.steps,rank=64,knots=129,action_dim=2,lr=3e-4,
        target_queries_per_update=512,defensive_uniform=.5,temperature=1.,
        target="GMM40-seed0-box-normalized",coordinate_scale=50.,
        loss="integral(exp(Q))-E_proposal[exp(Qtarget)/proposal*Q]",
        normalized_importance_weights=False,eval_samples=16384,eval_every=args.eval_every,
        phase=args.phase,attempt=args.attempt,
        initialization="target-independent 8x8 overlapping grid; leaf width .25 only at initialization; seed jitter",
        actor_network=None,critic_network="one trainable bounded sum-product spline circuit",
        learned_sigma=False,tanh=False,state_conditioner=False)
    state,key=initialize_circuit(args.seed,lr=cfg["lr"])
    target=target_parameters()
    evaluation=evaluation_set(target,cfg["eval_samples"])
    uniform=jax.random.uniform(jax.random.PRNGKey(170921),(4096,2),minval=-1.,maxval=1.)
    cfg["trainable_parameters"]=sum(v.size for v in jax.tree_util.tree_leaves(state.params))
    meta=dict(config=cfg,source_commit=sha,command=sys.argv,hostname=platform.node(),
        device=[str(d) for d in jax.devices()],slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),output=str(out))
    meta["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip()
    if not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("Main experiment must run on its allocated GPU")
    write_json(out/"manifest.json",meta)
    np.savez(out/"target.npz",centers=np.asarray(target[0]),log_std=np.asarray(target[1]))
    name=f"GMM40-SplineEnergy-{args.phase}-seed{args.seed}-{args.attempt}"
    run=wandb.init(entity="OptiQ",project=PROJECT,name=name,
        id=hashlib.sha256(str(out).encode()).hexdigest()[:12],resume="never",
        group=args.phase,job_type="train",dir=str(out),config=dict(cfg,source_commit=sha),
        tags=["GMM40","joint-Q-policy","spline-energy","one-step",args.phase],
        settings=wandb.Settings(init_timeout=120))
    write_json(out/"wandb.json",dict(id=run.id,url=run.url,project=PROJECT))
    run.define_metric("update")
    run.define_metric("*",step_metric="update")
    run.summary.update(meta)
    update=make_circuit_update(target)
    @jax.jit
    def advance(state,key):
        def step(carry,_):
            state,key,info=update(*carry)
            return (state,key),info
        (state,key),info=jax.lax.scan(step,(state,key),None,length=args.eval_every)
        return state,key,{k:jnp.mean(v) for k,v in info.items()}
    tick=time.monotonic()
    advance=advance.lower(state,key).compile()
    compilation_seconds=time.monotonic()-tick
    run.summary["compilation_seconds"]=compilation_seconds
    training_seconds=0.
    info={}
    start=time.monotonic()
    try:
        with (out/"history.jsonl").open("w") as history:
            for step in range(0,args.steps+1,args.eval_every):
                if step:
                    tick=time.monotonic()
                    state,key,info=advance(state,key)
                    jax.block_until_ready(state)
                    training_seconds+=time.monotonic()-tick
                metrics,sample,mass,refmass=evaluate(state,target,evaluation,uniform)
                scalar={k:float(v) for k,v in info.items()}
                row=dict(update=step,training_seconds=training_seconds,
                    wall_seconds=time.monotonic()-start,target_logp_evals=step*512,
                    target_score_evals=0,**metrics,**scalar)
                if not all(np.isfinite(v) for v in row.values()):
                    raise FloatingPointError("Nonfinite metric/state")
                history.write(json.dumps(row,allow_nan=False)+"\n"); history.flush()
                write_json(out/"latest.json",row)
                payload=dict(row)
                if step==0 or step%5000==0 or step==args.steps:
                    png=out/f"distribution_{step:06d}.png"
                    plot(png,target,sample,mass,refmass,f"{name} | step {step} | KL(p||q)={metrics['forward_kl']:.4f}")
                    payload["distribution"]=wandb.Image(str(png))
                run.log(payload,step=step)
                run.summary.update(row)
                if step==0 or step%5000==0 or step==args.steps:
                    (out/f"checkpoint_{step:06d}.msgpack").write_bytes(serialization.to_bytes(dict(state=state,key=key)))
                    np.savez_compressed(out/f"samples_{step:06d}.npz",samples=sample,mode_mass=mass,reference_mass=refmass)
                print(json.dumps(row),flush=True)
        timing=inference_benchmark(state,args.seed,out)
        run.summary["inference_timing"]=timing
        artifact=wandb.Artifact(name.lower(),type="gmm40-result",metadata={"source_commit":sha})
        for f in [out/"manifest.json",out/"history.jsonl",out/"inference_timing.json",
                  out/f"checkpoint_{args.steps:06d}.msgpack",out/f"samples_{args.steps:06d}.npz",
                  out/f"distribution_{args.steps:06d}.png"]:
            artifact.add_file(str(f))
        run.log_artifact(artifact)
        url=run.url
        run.summary["completed"]=True
        run.finish()
        write_json(out/"COMPLETE.json",dict(steps=args.steps,source_commit=sha,wandb_url=url,metrics=row))
    except BaseException as exc:
        write_json(out/"FAILED.json",dict(error=repr(exc),source_commit=sha))
        run.finish(exit_code=1)
        raise

if __name__=="__main__":
    main()

