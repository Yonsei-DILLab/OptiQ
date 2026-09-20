"""Persistent fixed-Q GMM40 training through the shared v7 RL actor update.

Fresh output directories only. Resume writes a new directory and retains the
parent's full actor/Adam/RNG checkpoint. This entry does not change the bounded
implementation-validation runner or any learning algorithm setting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import jax
import numpy as np

from .evaluation import atomic_json, save_evaluation
from .target import ROOT
from .validate_v7 import checkpoint_audit, make_agent, source_diagnostics


TRAINING_DEPENDENCIES = (
    'gmm40/v7.py', 'gmm40/target.py', 'gmm40/target_definition.json',
    'optiq_dime/conditional_sac.py', 'optiq_dime/latent_transport.py',
    'optiq_dime/persistent_transport.py',
    'optiq_dime/policy.py', 'optiq_dime/semi_implicit.py',
    'optiq_dime/transport.py', 'optiq_dime/optimizers.py',
    'optiq_dime/latent.py', 'models/utils.py',
)
SCIENTIFIC_FIELDS = (
    'seed', 'batch', 'num_students', 'proposal_components', 'proposals_per_component',
    'teacher_sampling_mode', 'actor_samples', 'teacher_resample_count', 'latent_seed',
    'proposal_std', 'temperature', 'epsilon', 'iterations', 'hidden_dims',
    'actor_learning_rate', 'actor_log_std_bounds', 'actor_initial_sigma',
    'actor_max_grad_norm',
    'source_importance_correction',
    'latent_prior_std', 'action_scale',
    'potential_solver', 'dual_learning_rate', 'dual_hidden_dims',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_config(seed, potential_solver='persistent_dual', epsilon=.1, actor_max_grad_norm=None,
                      source_importance_correction=True):
    return dict(seed=seed, batch=256, num_students=4096,
                proposal_components=256, proposals_per_component=1,
                teacher_sampling_mode='stratified', actor_samples=16,
                teacher_resample_count=16, latent_seed=seed,
                proposal_std=.05, temperature=1., epsilon=epsilon, iterations=100,
                hidden_dims=[256, 256], actor_learning_rate=3e-4,
                actor_max_grad_norm=actor_max_grad_norm,
                source_importance_correction=source_importance_correction,
                actor_log_std_bounds=[-5., 1.], actor_initial_sigma=.5,
                latent_prior_std=1., action_scale=40., potential_solver=potential_solver,
                dual_learning_rate=1e-4, dual_hidden_dims=[256, 256])


def verify_preflight(preflight, source_hashes, potential_solver=None, actor_max_grad_norm=None,
                     source_importance_correction=True):
    checks = json.loads(Path(preflight).read_text())
    assert checks['status'] == 'passed'
    assert checks['teacher_component_ids_each_once']
    assert checks['teacher_draws_per_component'] == 1
    if potential_solver is not None:
        assert checks.get('potential_solver', 'fresh_sinkhorn') == potential_solver
    if actor_max_grad_norm is not None:
        assert actor_max_grad_norm in checks.get('validated_actor_max_grad_norms', []), \
            'Requested actor clipping was not validated by this preflight'
    if not source_importance_correction:
        assert False in checks.get('validated_source_importance_correction', []), \
            'Disabled actor source correction was not validated by this preflight'
    for name, expected in checks['source_sha256'].items():
        assert source_hashes[name] == expected, f'Preflight source changed: {name}'
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--evaluate-every', type=int, default=5000)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--potential-solver', choices=('persistent_dual', 'fresh_sinkhorn'), default='persistent_dual')
    parser.add_argument('--epsilon', type=float, default=.1,
                        help='OT assignment epsilon; teacher temperature and SAC alpha remain 1')
    parser.add_argument('--actor-max-grad-norm', type=float,
                        help='Optional actor global L2 clipping BEFORE Adam; importance weights unchanged')
    parser.add_argument('--source-importance-correction', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Correct OT source sampling to uniform latent prior; disabling retains teacher W')
    args = parser.parse_args()
    if min(args.steps, args.chunk, args.evaluate_every) < 1:
        parser.error('Positive step, chunk and evaluation interval required')
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('Positive finite OT epsilon required')
    if args.actor_max_grad_norm is not None and (
            not np.isfinite(args.actor_max_grad_norm) or args.actor_max_grad_norm <= 0):
        parser.error('Actor max grad norm must be positive and finite')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = dict(scientific_config(args.seed, args.potential_solver, args.epsilon,
                                   args.actor_max_grad_norm, args.source_importance_correction),
                  requested_updates=args.steps,
                  steps=args.steps, chunk=args.chunk, evaluate_every=args.evaluate_every,
                  profile='v7_one_per_latent256_h4096_k16', scope='fixed_Q_training',
                  Q='log p_original(40 * normalized_action)', alpha=1.,
                  shared_update='optiq_dime.conditional_sac.update_actor',
                  fixed_student_latents=True, fresh_teacher_every_update=True,
                  fresh_teacher_and_OT_every_update=args.potential_solver == 'fresh_sinkhorn',
                  teacher_component_allocation='one sample from each of 256 independent fresh latent conditionals',
                  source_importance_self_normalized=False, source_importance_clipped=False,
                  actor_objective=(
                      'mean[((1/H)/sum_j P_ij)*(T log pi_i - Q - T log Pr(i|a,s))]'
                      if args.source_importance_correction else
                      'mean[T log pi_i - Q - T log Pr(i|a,s)] over OT-selected latents'),
                  actor_gradient_clipping_stage='pre_Adam_global_L2' if args.actor_max_grad_norm is not None else None,
                  auxiliary_dual_network=args.potential_solver == 'persistent_dual',
                  persistent_dual_state=args.potential_solver == 'persistent_dual',
                  dual_observation='ones[1,1]', dual_updates_per_actor=1 if args.potential_solver == 'persistent_dual' else 0,
                  performance_early_stop_enabled=False, speed_guard_enabled=False,
                  evaluation_samples=10000, evaluation_seed=900000+args.seed,
                  evaluation_reference_seed=20260917, cuda_visible=os.getenv('CUDA_VISIBLE_DEVICES'),
                  jax_version=jax.__version__, pid=os.getpid(), started_unix=time.time())
    atomic_json(out/'config.json', config)
    atomic_json(out/'status.json', dict(status='initializing', step=0, pid=os.getpid(), scope='fixed_Q_training'))
    agent = None
    start_updates = 0
    train_seconds = 0.
    warm_times, sizes_seen, audits, probes = [], set(), [], []
    stop = dict(signal=None)
    def request_stop(number, _frame):
        stop['signal'] = signal.Signals(number).name
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def runtime():
        return dict(training_seconds=train_seconds,
                    actual_updates=agent.updates if agent is not None else start_updates,
                    starting_update=start_updates,
                    updates_this_process=(agent.updates-start_updates) if agent is not None else 0,
                    median_warm_update_seconds=float(np.median(warm_times)) if warm_times else None,
                    last_warm_update_seconds=warm_times[-1] if warm_times else None,
                    warm_blocks=len(warm_times),
                    timing_note='First block per chunk size excluded; evaluation/checkpoint time excluded')

    def save_checkpoint(label=None):
        path = out/'checkpoints'/f'{label or "step"}_{agent.updates:07d}.bin'
        temporary = path.with_suffix('.bin.tmp')
        agent.save(temporary)
        audit = checkpoint_audit(temporary, agent.updates)
        temporary.replace(path)
        audits.append(dict(step=agent.updates, checkpoint=str(path), **audit))
        atomic_json(out/'checkpoint_audits.json', audits)
        return path

    try:
        (out/'checkpoints').mkdir()
        source = out/'source'
        for directory in ('gmm40', 'optiq_dime', 'common', 'models'):
            shutil.copytree(ROOT/directory, source/directory,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        config['source_sha256'] = {str(path.relative_to(source)): digest(path)
                                  for path in source.rglob('*') if path.is_file()}
        config['training_dependency_sha256'] = {name: config['source_sha256'][name]
                                               for name in TRAINING_DEPENDENCIES}
        checks = verify_preflight(args.preflight, config['source_sha256'], args.potential_solver,
                                  args.actor_max_grad_norm, args.source_importance_correction)
        config['preflight_validated_seed'] = checks['seed']
        config['preflight_path'] = str(args.preflight.resolve())
        atomic_json(out/'integration_checks.json', checks)
        agent = make_agent(config)
        if args.resume_checkpoint:
            checkpoint = args.resume_checkpoint.resolve()
            parent = checkpoint.parent.parent
            parent_config = json.loads((parent/'config.json').read_text())
            for field in SCIENTIFIC_FIELDS:
                if field == 'actor_max_grad_norm':
                    previous = parent_config.get(field)
                elif field == 'source_importance_correction':
                    previous = parent_config.get(field, True)
                else:
                    previous = parent_config[field]
                assert previous == config[field], f'Resume setting differs: {field}'
            for name, current in config['training_dependency_sha256'].items():
                assert parent_config['source_sha256'][name] == current, f'Resume training code differs: {name}'
            agent.restore(checkpoint)
            checkpoint_audit(checkpoint, agent.updates)
            start_updates = agent.updates
            if start_updates >= args.steps:
                raise ValueError('Requested final update must exceed checkpoint update')
            config['resume'] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
                                    parent_directory=str(parent), starting_update=start_updates)
        else:
            config['resume'] = None
        config['settings_signature'] = agent.settings_signature
        atomic_json(out/'config.json', config)
        reference = agent.target.sample(10000, config['evaluation_reference_seed'])
        full_reference = agent.target.sample(10000, config['evaluation_reference_seed'], bounded=False)
        goals = sorted({start_updates, args.steps, *range(args.evaluate_every, args.steps+1, args.evaluate_every),
                        *[step for step in (1000, 5000, 10000, 25000, 50000, 75000, 100000)
                          if start_updates <= step <= args.steps]})
        goals = [step for step in goals if start_updates <= step <= args.steps]
        info = {}
        for goal in goals:
            while agent.updates < goal and stop['signal'] is None:
                count = min(args.chunk, goal-agent.updates)
                started = time.monotonic()
                info = agent.advance(count)
                elapsed = time.monotonic()-started
                train_seconds += elapsed
                assert int(agent.state.step) == agent.updates
                if agent.dual_state is not None:
                    assert int(agent.dual_state.step) == agent.updates
                if count in sizes_seen:
                    warm_times.append(elapsed/count)
                sizes_seen.add(count)
                if not all(np.isfinite(value) for value in info.values()):
                    raise FloatingPointError(f'Nonfinite metrics at update {agent.updates}: {info}')
                assert info['actor_nll_used'] == 0
                assert info['actor_source_importance_used'] == float(args.source_importance_correction)
                assert info['ot_fresh_solve'] == float(args.potential_solver == 'fresh_sinkhorn')
                if args.potential_solver == 'persistent_dual':
                    assert info['ot_persistent_dual'] == info['ot_dual_updates'] == 1
                assert info['teacher_one_per_latent'] == 1
                assert info['teacher_proposal_component_count'] == 256
                assert info['actor_training_pairs'] == info['ot_teacher_count'] == 16
                row = dict(step=agent.updates, train_seconds=train_seconds,
                           block_seconds=elapsed, updates_in_block=count, metrics=info)
                with (out/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                atomic_json(out/'runtime.json', runtime())
                atomic_json(out/'status.json', dict(status='training', scope='fixed_Q_training',
                            pid=os.getpid(), requested_updates=args.steps, **row))
            checkpoint = save_checkpoint()
            step = agent.updates
            evaluation_key = np.asarray(agent.key).copy()
            samples, extra = agent.evaluate_samples(10000, config['evaluation_seed'])
            diagnostic = source_diagnostics(agent)
            assert np.array_equal(evaluation_key, np.asarray(agent.key))
            probes.append(dict(step=step, **diagnostic))
            atomic_json(out/'source_diagnostics.json', probes)
            result = save_evaluation(out, f'v7 one-per-latent256 seed {args.seed}', step,
                        samples, agent.target, reference, full_reference,
                        dict(info, train_seconds=train_seconds), extra_samples=extra)
            print(json.dumps(dict(event='evaluation', seed=args.seed, step=step,
                near=result['high_density_fraction'], coverage=result['mode_coverage'],
                mmd2=result['mmd2'], checkpoint=str(checkpoint), runtime=runtime()), allow_nan=False), flush=True)
            if stop['signal'] is not None:
                break
        completed = agent.updates == args.steps
        atomic_json(out/'runtime.json', runtime())
        atomic_json(out/'completion.json', dict(status='completed' if completed else 'stopped',
                    requested_updates=args.steps, actual_updates=agent.updates,
                    saved_actor_Adam_counts_verified=True, final_metrics=result,
                    saved_dual_Adam_counts_verified=agent.dual_state is not None,
                    stop_signal=stop['signal'], success_of_distribution_recovery_claimed=False))
        atomic_json(out/'status.json', dict(status='completed' if completed else 'stopped',
                    scope='fixed_Q_training', step=agent.updates, actor_updates=int(agent.state.step),
                    requested_updates=args.steps, pid=os.getpid(), train_seconds=train_seconds,
                    stop_signal=stop['signal']))
    except Exception as error:
        failure_checkpoint = None
        if agent is not None:
            failure_checkpoint = save_checkpoint('failure')
        atomic_json(out/'runtime.json', runtime())
        atomic_json(out/'status.json', dict(status='failed', scope='fixed_Q_training',
                    error=repr(error), step=agent.updates if agent else start_updates,
                    requested_updates=args.steps, pid=os.getpid(),
                    failure_checkpoint=str(failure_checkpoint) if failure_checkpoint else None))
        raise


if __name__ == '__main__':
    main()
