"""Train g(z) with the current OptiQ update and exact GMM log density."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone
import uuid
from contextlib import nullcontext

import flax.serialization
import jax
import numpy as np
import torch
import wandb

from optiq_dime.runtime import load_environment, provenance, ROOT
from .metrics import evaluate_sample_tensor
from .sampler import default_config, initialize, make_target, update, draw, training_candidates_per_update, scheduled_config
from .accounting import density_evaluations
from .training_clock import TrainingClock


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/gmm40/runs')
    parser.add_argument('--job-type', default='train')
    parser.add_argument('--group', default='gmm40-current-optiq-T0.25-beta0.1-30k')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--resume-checkpoint', type=Path, help='Continue an actor/optimizer checkpoint, retaining its update count')
    for name, value in default_config().items():
        if name == 'hidden_dims':
            parser.add_argument('--hidden-dims', type=int, nargs='+', default=value)
            continue
        if isinstance(value, bool):
            parser.add_argument('--' + name.replace('_', '-'), action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    cfg = {k: getattr(args, k) for k in default_config() if k != 'hidden_dims'}
    cfg['hidden_dims'] = args.hidden_dims
    if any(width <= 0 for width in cfg['hidden_dims']):
        parser.error('Hidden dimensions must be positive')
    if cfg['latent_sampling'] not in ('iid','grid'):
        parser.error('latent-sampling must be iid or grid')
    if cfg['initialization'] not in ('random','identity','identity_wide'):
        parser.error('initialization must be random, identity or identity_wide')
    if cfg['initialization'] in ('identity','identity_wide') and (cfg['initial_output_std'] <= 0 or
            cfg['initialization_noise'] < 0 or min(cfg['hidden_dims']) < 4):
        parser.error('Identity initialization requires positive initial-output-std, nonnegative initialization-noise and widths>=4')
    if cfg['initialization'] == 'identity_wide' and any(w % 2 for w in cfg['hidden_dims']):
        parser.error('Wide identity initialization requires even hidden widths')
    cfg.update(seed=args.seed, target_seed=0, oracle='exact log p(x), duplicated heads; no learned critic',
        sampler=f'x={cfg["coordinate_scale"]}*clip(ImplicitActor(empty observation,z),-1,1)',
        temperature_schedule='constant', proposal_sampling_mode='stratified',
        transport_target_mode='argmax', distillation_loss='pointwise_mse',
        interpretation=f"Target score is log p(x)/T - beta*log q_KDE(x), with final T={cfg['temperature']} and beta={cfg['density_beta']}. Finite OT and neural distillation do not guarantee exact p sampling.")
    if cfg['anneal_updates'] > 0 and cfg['temperature_start'] > 0:
        cfg['temperature_schedule'] = 'exponential, then constant'
    for k in ['updates', 'batch_size', 'num_policy_samples', 'proposals_per_policy_sample', 'eval_interval', 'eval_samples', 'checkpoint_interval', 'log_interval']:
        if cfg[k] <= 0:
            parser.error(k + ' must be positive')
    if cfg['eval_samples'] < 2 or cfg['temperature'] <= 0 or cfg['coordinate_scale'] <= 0:
        parser.error('Invalid sample count, temperature or coordinate scale')
    if cfg['proposals_per_policy_sample'] <= int(cfg['include_anchor']):
        parser.error('At least one random candidate per KDE center is required; with --include-anchor use proposals-per-policy-sample >= 2')
    cfg['proposal_std_physical'] = cfg['proposal_std'] * cfg['coordinate_scale']
    cfg['proposal_variance_physical'] = cfg['proposal_std_physical'] ** 2
    cfg['proposal_distribution'] = 'gaussian_unbounded' if cfg['unbounded_actions'] else 'gaussian_truncated'
    cfg['effective_proposal_clip_physical'] = None if cfg['unbounded_actions'] else cfg['proposal_clip'] * cfg['coordinate_scale']
    cfg['effective_output_bound_physical'] = None if cfg['unbounded_actions'] else cfg['coordinate_scale']
    cfg['coordinate_system'] = 'native' if cfg['coordinate_scale'] == 1. else 'scaled'
    if cfg['unbounded_actions']:
        prefix = '' if cfg['coordinate_scale'] == 1. else f'{cfg["coordinate_scale"]}*'
        cfg['sampler'] = f'x={prefix}ImplicitActor(empty observation,z), without clipping'
    cfg['effective_kde_center_count'] = cfg['num_policy_samples']
    cfg['effective_candidate_count'] = cfg['num_policy_samples'] * cfg['proposals_per_policy_sample']
    cfg['anchor_candidate_count'] = cfg['num_policy_samples'] * int(cfg['include_anchor'])
    cfg['random_candidate_count'] = cfg['effective_candidate_count'] - cfg['anchor_candidate_count']
    if cfg['include_anchor']:
        cfg['interpretation'] += ' Anchors are deterministic KDE centers; the unchanged legacy rule also scores them with continuous KDE density, not an exact mixed-measure importance correction.'
    cfg['shared_unconditional_candidate_cloud'] = False
    cfg['training_density_evaluations_per_update'] = training_candidates_per_update(cfg)
    cfg['group'] = args.group
    if args.resume_checkpoint:
        parent_config = json.loads((args.resume_checkpoint.parent / 'config.json').read_text())
        for field in ('seed', 'hidden_dims', 'coordinate_scale', 'unbounded_actions'):
            if parent_config.get(field) != cfg[field]:
                raise ValueError(f'Resume configuration mismatch: {field}')
        if parent_config.get('initial_output_std', 0.) != cfg['initial_output_std']:
            raise ValueError('Resume must retain the recorded initialization setting')
        if parent_config.get('initialization', 'random') != cfg['initialization']:
            raise ValueError('Resume must retain the recorded initialization type')
        if cfg['initialization'] in ('identity','identity_wide') and parent_config.get('initialization_noise', .01) != cfg['initialization_noise']:
            raise ValueError('Resume must retain the recorded initialization noise')
        cfg['resume_checkpoint'] = str(args.resume_checkpoint.resolve())
        cfg['resume_checkpoint_sha256'] = hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest()
        cfg['parent_wandb_url'] = parent_config.get('wandb_url')
    load_environment()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f'gmm40-K{cfg["effective_kde_center_count"]}-M{cfg["effective_candidate_count"]}-anchor{int(cfg["include_anchor"])}-unbounded{int(cfg["unbounded_actions"])}-scale{cfg["coordinate_scale"]}-std{cfg["proposal_std_physical"]}-T{args.temperature}-beta{args.density_beta}-seed{args.seed}-{stamp}-{uuid.uuid4().hex[:6]}'
    output = args.output_root.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', name=name, group=args.group,
        job_type=args.job_type, tags=['gmm40', 'anchor' if cfg['include_anchor'] else 'no-anchor', 'unbounded' if cfg['unbounded_actions'] else 'bounded', 'g(z)', 'fixed-temperature' if cfg['temperature_schedule']=='constant' else 'annealed-temperature'],
        config=cfg, dir=str(output), save_code=False)
    step = 0
    actor = None
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        runtime = provenance()
        if args.require_gpu and runtime['jax_backend'] != 'gpu':
            raise RuntimeError('GPU required but JAX selected ' + runtime['jax_backend'])
        files = ['optiq_dime/algorithm.py', 'optiq_dime/policy.py', 'optiq_dime/transport.py',
            'benchmarks/gmm40/sampler.py', 'benchmarks/gmm40/train.py', 'benchmarks/gmm40/metrics.py',
            'benchmarks/gmm40/target_torch.py', 'benchmarks/gmm40/latent_sampling.py',
            'benchmarks/gmm40/accounting.py', 'benchmarks/gmm40/training_clock.py']
        runtime['source_sha256'] = {}
        for file in files:
            data = (ROOT / file).read_bytes()
            runtime['source_sha256'][file] = hashlib.sha256(data).hexdigest()
            dest = output / 'source' / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        run.config.update({'runtime': runtime}, allow_val_change=True)
        write_json(output / 'config.json', dict(cfg, runtime=runtime, wandb_url=run.url))
        write_json(output / 'run.json', {'wandb_url': run.url, 'output': str(output)})
        print(json.dumps({'wandb_url': run.url, 'output': str(output)}), flush=True)
        actor, oracle, key = initialize(args.seed, cfg['hidden_dims'], cfg['learning_rate'], cfg['coordinate_scale'],
            cfg['initial_output_std'], cfg['initialization'], cfg['initialization_noise'])
        if args.resume_checkpoint:
            actor = flax.serialization.from_bytes(actor, args.resume_checkpoint.read_bytes())
            step = int(actor.step)
            if step >= cfg['updates']:
                raise ValueError('Requested final update must exceed checkpoint update')
            # The unchanged actor update advances this key by split(key, 4)[0].
            key = jax.lax.fori_loop(0, step, lambda _, k: jax.random.split(k, 4)[0], key)
        start_update = step
        cfg['start_update'] = start_update
        cfg['target_density_evaluations_at_start'] = (
            density_evaluations(parent_config, start_update) if args.resume_checkpoint else 0)
        run.config.update({k: cfg[k] for k in
            ('start_update', 'target_density_evaluations_at_start')})
        write_json(output / 'config.json', dict(cfg, runtime=runtime, wandb_url=run.url))
        target = make_target()
        # Evaluation RNG is completely separate from the training RNG.
        eval_key = jax.random.PRNGKey(1_000_000 + args.seed)
        history = output / 'history.jsonl'
        training_clock = None
        def evaluate():
            # Complete queued optimization before charging time to evaluation.
            # Evaluation already consumes these parameters; no per-update sync.
            jax.block_until_ready(actor.params)
            with training_clock.evaluation() if training_clock else nullcontext():
                samples = np.asarray(draw(actor, eval_key, cfg['eval_samples'], cfg['coordinate_scale'], cfg['unbounded_actions']))
                if not np.isfinite(samples).all():
                    raise FloatingPointError('Nonfinite generator samples')
                metrics = evaluate_sample_tensor(torch.from_numpy(samples.copy()), target, cfg['reference_seed'])
                metrics.update(update=step, elapsed_seconds=time.monotonic() - started,
                    target_density_evaluations=density_evaluations(cfg, step))
                np.save(output / 'samples_latest.npy', samples)
                write_json(output / 'latest_evaluation.json', metrics)
                with history.open('a') as f:
                    f.write(json.dumps(metrics, allow_nan=False) + '\n')
                run.log(metrics)
                print(json.dumps(metrics), flush=True)
            return metrics
        evaluate()
        setup_and_initial_evaluation_seconds = time.monotonic() - started
        training_clock = TrainingClock()
        def record_timing(checkpoint):
            # Checkpoint serialization has already synchronized the actor state.
            record = dict(update=step, start_update=start_update, checkpoint=checkpoint,
                setup_and_initial_evaluation_seconds=setup_and_initial_evaluation_seconds,
                **training_clock.snapshot())
            with (output / 'training_timing.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            write_json(output / 'latest_timing.json', record)
            return record
        for step in range(start_update + 1, cfg['updates'] + 1):
            effective_cfg = scheduled_config(cfg, step - 1)
            actor, loss, key, metrics = update(actor, oracle, key, effective_cfg)
            if step % cfg['log_interval'] == 0 or step == 1:
                loss = float(loss)
                if not np.isfinite(loss):
                    raise FloatingPointError('Nonfinite actor loss')
                names = ['source_ess_absolute', 'max_source_weight', 'policy_spread_l2', 'selected_delta_l2', 'source_q_std']
                if cfg['include_anchor']:
                    names += ['source_ess_fraction', 'local_anchor_argmax_fraction', 'local_best_q_gain_over_anchor', 'local_improvement_fraction']
                record = {f'train/{k}': float(metrics[k]) for k in names}
                record.update(update=step, actor_loss=loss, temperature=effective_cfg['temperature'],
                    proposal_std=effective_cfg['proposal_std'], sinkhorn_epsilon=effective_cfg['sinkhorn_epsilon'],
                    elapsed_seconds=time.monotonic()-started)
                with history.open('a') as f:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                run.log(record)
            if step % cfg['checkpoint_interval'] == 0:
                (output / f'actor_state_{step}.msgpack').write_bytes(flax.serialization.to_bytes(actor))
                record_timing(f'actor_state_{step}.msgpack')
            if step % cfg['eval_interval'] == 0:
                final_metrics = evaluate()
            if stop_requested:
                break
        final_metrics = evaluate()
        np.save(output / 'samples_final.npy', np.load(output / 'samples_latest.npy'))
        (output / 'actor_state_final.msgpack').write_bytes(flax.serialization.to_bytes(actor))
        timing = record_timing('actor_state_final.msgpack')
        status = dict(updates=step, finished=step == cfg['updates'], stopped=stop_requested,
            start_update=start_update, updates_executed=step-start_update,
            target_density_evaluations=density_evaluations(cfg, step),
            wandb_url=run.url, elapsed_seconds=time.monotonic()-started, final_metrics=final_metrics,
            timing=timing)
        write_json(output / ('completed.json' if status['finished'] else 'stopped.json'), status)
        run.summary.update(status)
        artifact = wandb.Artifact(name + '-results', type='gmm40-sampler')
        for file in ['config.json', 'samples_final.npy', 'actor_state_final.msgpack', 'history.jsonl', 'latest_evaluation.json',
                     'training_timing.jsonl', 'latest_timing.json']:
            artifact.add_file(str(output / file), name=file)
        artifact.add_dir(str(output / 'source'), name='source')
        run.log_artifact(artifact)
        run.finish(exit_code=0 if status['finished'] else 130)
    except BaseException as exc:
        if actor is not None:
            (output / 'actor_state_interrupted.msgpack').write_bytes(flax.serialization.to_bytes(actor))
        write_json(output / 'failed.json', {'update': step, 'error': repr(exc), 'wandb_url': run.url})
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
