"""One sequentially scheduled fixed-Q run, with frequent durable evaluations."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
from .target import ROOT,RESULTS,Target,initialize_target
from .evaluation import atomic_json,save_evaluation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",choices=["optiq","optiq_trg","sac","dipo","meow","mfpo","sql"],required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("--steps",type=int,default=100000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--n",type=int,default=16)
    parser.add_argument("--m",type=int,default=64)
    parser.add_argument("--batch",type=int,default=256)
    parser.add_argument("--epsilon",type=float,default=.1)
    parser.add_argument("--sinkhorn-iterations",type=int,default=100)
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--depth",type=int,default=2)
    parser.add_argument("--mean-output-init-scale",type=float,default=None,
                        help="Optional OptiQ fixed-Q mean-head variance scale; default 1e-4")
    parser.add_argument("--trg-log-std-max",type=float,default=-1.,
                        help="GMM40 TRG-only sigma ablation; lower bound remains -5")
    parser.add_argument("--trg-initial-log-std",type=float,default=-1.,
                        help="GMM40 TRG-only initial log sigma, within the configured bounds")
    parser.add_argument("--trg-teacher-std-floor",type=float,default=math.exp(-5),
                        help="GMM40 TRG teacher-only std floor, shared by sampling and proposal density")
    parser.add_argument("--nll-top-k",type=int,default=None,
                        help="Fixed-Q OptiQ only: retain this many highest-mass OT targets per row in NLL")
    parser.add_argument("--nll-plan-threshold",type=float,default=None,
                        help="Fixed-Q OptiQ only: discard raw OT P entries <= threshold and renormalize rows")
    parser.add_argument("--optiq-devices",type=int,default=1,
                        help="Fixed-Q OptiQ only: synchronous data-parallel devices; batch is the global batch")
    parser.add_argument("--sigma-row-balance",action="store_true",
                        help="Fixed-Q full-row OptiQ: balance only the log-sigma gradient across rows")
    parser.add_argument("--eval-samples",type=int,default=10000)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--navigation",action="store_true")
    parser.add_argument("--warmup",type=int,default=5000)
    parser.add_argument("--eval-episodes",type=int,default=1000)
    parser.add_argument("--temperature",type=float,default=None)
    parser.add_argument("--sql-kernel-particles",type=int,default=16)
    parser.add_argument("--sql-kernel-update-ratio",type=float,default=.5)
    parser.add_argument("--sql-value-particles",type=int,default=16)
    parser.add_argument("--sql-target-update-interval",type=int,default=1000)
    args=parser.parse_args()
    if (args.trg_log_std_max, args.trg_initial_log_std, args.trg_teacher_std_floor) != (-1., -1., math.exp(-5)) and (
            args.method != 'optiq_trg' or args.navigation or args.resume):
        raise ValueError('TRG sigma overrides require a fresh fixed-Q optiq_trg run')
    if args.method == 'optiq_trg' and not (
            math.isfinite(args.trg_log_std_max) and math.isfinite(args.trg_initial_log_std)
            and -5. < args.trg_log_std_max and -5. <= args.trg_initial_log_std <= args.trg_log_std_max):
        raise ValueError('TRG requires -5 < log_std_max and initial log std within bounds')
    if not math.isfinite(args.trg_teacher_std_floor) or args.trg_teacher_std_floor <= 0:
        raise ValueError('Teacher std floor must be positive and finite')
    if args.optiq_devices<1 or (args.optiq_devices!=1 and
            (args.method!='optiq' or args.navigation or args.batch%args.optiq_devices)):
        raise ValueError('Parallel devices require fixed-Q OptiQ and an evenly divisible global batch')
    if args.nll_top_k is not None and (args.method!='optiq' or args.navigation or not 1<=args.nll_top_k<=args.m):
        raise ValueError('nll_top_k requires fixed-Q OptiQ and 1 <= K <= m')
    if args.nll_plan_threshold is not None and (args.method!='optiq' or args.navigation or
            not math.isfinite(args.nll_plan_threshold) or args.nll_plan_threshold<=0 or args.nll_top_k is not None):
        raise ValueError('nll_plan_threshold requires fixed-Q OptiQ, a positive finite threshold, and no top-K selection')
    if args.sigma_row_balance and (args.method!='optiq' or args.navigation or
            args.nll_top_k is not None or args.nll_plan_threshold is not None):
        raise ValueError('sigma_row_balance requires fixed-Q OptiQ with the original full-row NLL')
    explicit_actor_hparams = args.mean_output_init_scale is not None
    if explicit_actor_hparams and (args.method != 'optiq' or args.navigation or args.resume):
        raise ValueError('Actor initialization override currently requires a fresh fixed-Q OptiQ run')
    if args.temperature is None:
        args.temperature=.25 if args.navigation and args.method=='optiq' else 1.0
    if not math.isfinite(args.temperature) or args.temperature<=0:
        raise ValueError("Temperature must be positive and finite")
    if args.method=='sql':
        from .sql import config_from_args
        config_from_args(args)
        if args.batch<=0:raise ValueError('SQL batch must be positive')
        if args.navigation and args.resume:
            raise ValueError('Navigation CLI resume is not supported; SQLOnline.restore restores the learner/replay only')
    if args.navigation:
        if args.method=='optiq_trg':raise ValueError('optiq_trg is a fixed-Q adapter; use the TRG RL trainer for navigation')
        from .navigation_run import run_navigation
        run_navigation(args)
        return
    if args.temperature!=1.0 and args.method not in ('optiq','optiq_trg','sql'):
        raise ValueError("Fixed-Q temperature controls are implemented only for OptiQ and SQL")
    if args.resume and args.method=='optiq':
        parent_config=json.loads((args.resume.parent.parent/'config.json').read_text())
        if parent_config.get('temperature',1.0)!=args.temperature:
            raise ValueError('Resume temperature must match the saved run; use a fresh run for a temperature control')
        if parent_config.get('nll_top_k')!=args.nll_top_k:
            raise ValueError('Resume NLL top-K must match the saved run; use a fresh run for this ablation')
        if parent_config.get('nll_plan_threshold')!=args.nll_plan_threshold:
            raise ValueError('Resume NLL plan threshold must match the saved run; use a fresh run for this ablation')
        if parent_config.get('optiq_devices',1)!=args.optiq_devices:
            raise ValueError('Resume device count must match the saved run')
        if parent_config.get('sigma_row_balance',False)!=args.sigma_row_balance:
            raise ValueError('Resume sigma row balancing must match the saved run')
    folder=RESULTS/args.name
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"checkpoints").mkdir()
    initialize_target()
    target=Target()
    # References must never enter an adapter constructor or an update call.
    reference=target.sample(args.eval_samples,20260917,bounded=True)
    full_reference=target.sample(args.eval_samples,20260917,bounded=False)
    config=vars(args).copy(); config["resume"]=str(args.resume) if args.resume else None
    if args.method=='optiq_trg':
        config.update(actor_learning_rate=3e-4,actor_log_std_bounds=[-5.,args.trg_log_std_max],
                      initial_log_std=args.trg_initial_log_std,
                      mean_output_init_scale=1.,
                      latent_mode='random',density_beta=1.,teacher_std_floor=args.trg_teacher_std_floor,
                      loss='direct marginal box-truncated Gaussian mixture NLL',
                      implementation='analysis_tools/experiments/20260920_truncated_mll/optiq_dime')
    config.update(source_git_commit=os.getenv('GMM40_SOURCE_COMMIT'),campaign=os.getenv('GMM40_CAMPAIGN'))
    if args.method=='sql':
        from .sql import metadata
        config.update(metadata(args))
    if args.method == 'optiq':
        config.update(actor_learning_rate=3e-4,
                      mean_output_init_scale=1e-4 if args.mean_output_init_scale is None else args.mean_output_init_scale,
                      nll_target_selection='full_row' if args.nll_top_k is None else 'top_k_by_ot_mass_renormalized')
        if args.nll_plan_threshold is not None:
            config.update(nll_target_selection='raw_P_strict_threshold_renormalized',
                          nll_empty_row_policy='original_full_row_fallback',
                          teacher_std_floor=.05,actor_log_std_bounds=[-5.,1.])
        if args.optiq_devices>1:
            config.update(global_batch=args.batch,local_batch=args.batch//args.optiq_devices,
                          gradient_aggregation='Mean across devices before one Adam update',
                          rng_sharding='fold_in(initial learner key, device index); independent per-device streams')
        if args.sigma_row_balance:
            config.update(
                sigma_row_weight_formula='rho_i=mean_d(E_row[(u_jd-mu_id)^2]/sigma_id^2); a_i=1/max(1,rho_i); w_i=stop_gradient(a_i/mean_rows(a_i))',
                sigma_row_weight_normalization='Independently per OT cloud; weights have mean 1 over N rows',
                gradient_intervention='NLL forward value and direct mu gradient unchanged; direct log_std gradient multiplied by w_i',
                shared_trunk_caveat='Sigma-path gradients into shared layers change; subsequent mean outputs are not constrained to match baseline')
    config.update(Q="log p_original(x)",temperature=args.temperature,seed=args.seed,
                  fixed_q_temperature_control=args.temperature!=1.0,
                  teacher_boltzmann_power=1.0/args.temperature,
                  evaluation_target_temperature=1.0,
                  target_sha256=hashlib.sha256((RESULTS/"target/definition.json").read_bytes()).hexdigest(),
                  python=platform.python_version(),pid=os.getpid(),cuda_visible=os.getenv("CUDA_VISIBLE_DEVICES"),
                  coordinate_convention=("physical x=40*a, native normalized bounded-action sampler" if args.method in ("dipo","mfpo") else "physical x=40*tanh(u); OT distance uses normalized x/40"),
                  training_data="No ground-truth samples; fixed energy queries only",
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))})
    atomic_json(folder/"config.json",config)
    if args.method=='optiq_trg':
        config['coordinate_convention']='physical x=40*a; box-truncated Gaussian in normalized a with center=tanh(raw_mu)'
        atomic_json(folder/"config.json",config)
    atomic_json(folder/"status.json",dict(status="initializing",pid=os.getpid(),step=0))
    try:
        if args.method=="optiq":
            from .optiq import OptiQ
            implementation=OptiQ;parallel_args={}
            if args.optiq_devices>1:
                from .optiq_parallel import ParallelOptiQ
                implementation=ParallelOptiQ;parallel_args={'devices':args.optiq_devices}
            agent=implementation(target,args.seed,args.n,args.m,args.batch,args.epsilon,(args.width,)*args.depth,args.sinkhorn_iterations,
                         mean_output_init_scale=config['mean_output_init_scale'],temperature=args.temperature,
                         nll_top_k=args.nll_top_k,nll_plan_threshold=args.nll_plan_threshold,
                         sigma_row_balance=args.sigma_row_balance,**parallel_args)
        elif args.method=="optiq_trg":
            from .optiq_trg import OptiQTRG
            agent=OptiQTRG(target,args.seed,args.n,args.m,args.batch,(args.width,)*args.depth,args.temperature,
                           log_std_max=args.trg_log_std_max,initial_log_std=args.trg_initial_log_std,
                           teacher_std_floor=args.trg_teacher_std_floor)
        elif args.method=="sql":
            from .sql import SQL,config_from_args
            agent=SQL(target,args.seed,args.batch,config_from_args(args))
        elif args.method=="mfpo":
            from .mfpo import MFPO
            agent=MFPO(target,args.seed,args.batch)
        else:
            from .torch_agents import make_agent
            agent=make_agent(args.method,target,args.seed,args.batch)
        if args.resume: agent.restore(args.resume)
        from .model_sizes import fixed_sizes
        fixed_sizes(folder,args.method,agent)
        step=agent.updates; training_seconds=0.; info={}
        if args.resume:
            parent_status=args.resume.parent.parent/"status.json"
            if parent_status.exists(): training_seconds=json.loads(parent_status.read_text()).get("train_seconds",0.)
        schedule=[100,500,1000,2500,5000]+list(range(10000,100001,10000))
        checkpoints=sorted(set([step]+[s for s in schedule if step<s<=args.steps]+[args.steps]))
        for goal in checkpoints:
            while step<goal:
                count=min(50,goal-step)
                started=time.monotonic()
                info=agent.advance(count)
                training_seconds+=time.monotonic()-started
                step=agent.updates
                if not all(np.isfinite(v) for v in info.values()): raise FloatingPointError(info)
                atomic_json(folder/"status.json",dict(status="training",pid=os.getpid(),step=step,train_seconds=training_seconds,metrics=info))
                if step%500==0: print(json.dumps(dict(event="train",name=args.name,step=step,train_seconds=training_seconds,metrics=info)),flush=True)
            agent.save(folder/"checkpoints"/f"step_{step:07d}.bin")
            started=time.monotonic()
            samples,rollout,extra=agent.evaluate_samples(args.eval_samples,900000+args.seed)
            sampling_seconds=time.monotonic()-started
            info.update(train_seconds=training_seconds,sampling_seconds=sampling_seconds)
            result=save_evaluation(folder,args.name,step,samples,target,reference,full_reference,info,rollout,extra)
            print(json.dumps(dict(event="evaluation",name=args.name,step=step,modes=result["mode_coverage"],mmd2=result["mmd2"],sw2=result["sliced_wasserstein2"],precision=result["high_density_fraction"])),flush=True)
        atomic_json(folder/"status.json",dict(status="completed",pid=os.getpid(),step=step,train_seconds=training_seconds))
        from .audit_updates import audit
        count_audit=audit(folder)
        if count_audit['errors']:raise RuntimeError(count_audit['errors'])
        if not args.name.startswith('validation_'):
            from .report import refresh
            refresh()
    except Exception as exc:
        atomic_json(folder/"status.json",dict(status="failed",pid=os.getpid(),error=repr(exc)))
        raise


if __name__=="__main__": main()
