"""Short GMM40 implementation validation through the shared v7 RL actor core.

This is a bounded smoke run, not a distribution-convergence experiment.
Use a fresh --out directory. Fixed H4096, 256 independent proposal latents,
one teacher draw per component, teacher resampling 256→16, 16 actor pairs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from optiq_dime import conditional_sac
from optiq_dime.persistent_transport import update_actor_persistent
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from .evaluation import atomic_json, save_evaluation
from .target import ROOT
from .v7 import GMM40V7


def tree_error(left, right):
    a, b = jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    assert len(a) == len(b)
    differences = []
    for x, y in zip(a, b):
        if isinstance(x, str):
            assert x == y
            continue
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        differences.append(float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    return max(differences, default=0.)


def checkpoint_audit(path, step):
    payload = Path(path).read_bytes()
    saved = flax.serialization.msgpack_restore(payload)
    counts = []
    def visit(tree):
        if isinstance(tree, dict):
            for name, value in tree.items():
                if name == 'count' and np.asarray(value).shape == ():
                    counts.append(int(value))
                else:
                    visit(value)
        elif isinstance(tree, (tuple, list)):
            for value in tree:
                visit(value)
    visit(saved['state']['opt_state'])
    assert int(saved['updates']) == int(saved['state']['step']) == step
    assert counts and set(counts) == {step}
    actor_counts = counts.copy()
    dual_counts = []
    if 'dual_state' in saved:
        assert int(saved['dual_state']['step']) == step
        counts.clear()
        visit(saved['dual_state']['opt_state'])
        dual_counts = counts.copy()
        assert dual_counts and set(dual_counts) == {step}
    assert np.asarray(saved['key']).shape == (2,)
    assert json.loads(saved['settings_signature'])['latent_seed'] is not None
    return dict(status='passed', saved_updates=int(saved['updates']), actor_updates=int(saved['state']['step']),
                optimizer_counts=actor_counts, training_rng_present=True,
                dual_updates=step if 'dual_state' in saved else None,
                dual_optimizer_counts=dual_counts,
                checkpoint_sha256=hashlib.sha256(payload).hexdigest())


def integration_checks(seed, potential_solver='persistent_dual'):
    """Compare this wrapper with the public function used by RL, plus resume."""
    agent = GMM40V7(seed=seed, batch=2, latent_seed=seed, potential_solver=potential_solver)
    if potential_solver == 'persistent_dual':
        direct = jax.jit(lambda state, dual, key: update_actor_persistent(
            state, dual, agent.observations, key, agent.q_fn,
            dual_observations=agent.dual_observations, **agent.settings))
        expected_state, expected_dual, expected_loss, expected_key, _ = direct(
            agent.state, agent.dual_state, agent.key)
    else:
        direct = jax.jit(lambda state, key: conditional_sac.update_actor(
            state, agent.observations, key, agent.q_fn, **agent.settings))
        expected_state, expected_loss, expected_key, _ = direct(agent.state, agent.key)
        expected_dual = None
    info = agent.advance(1)
    state_error = tree_error(agent.state, expected_state)
    key_error = tree_error(agent.key, expected_key)
    dual_error = tree_error(agent.dual_state, expected_dual)
    assert state_error < 2e-6 and key_error == 0, (state_error, key_error)
    assert dual_error < 2e-6, dual_error
    assert abs(info['loss']-float(expected_loss)) < 2e-5
    probe = GMM40V7(seed=seed, batch=2, latent_seed=seed, potential_solver=potential_solver)
    fresh_probe = GMM40V7(seed=seed, batch=2, latent_seed=seed, potential_solver='fresh_sinkhorn')
    assert tree_error(probe.state, fresh_probe.state) == 0
    assert tree_error(probe.key, fresh_probe.key) == 0
    proposal_keys = jax.random.split(probe.key, 7)
    first = probe.prepare()
    fresh_data = fresh_probe.prepare()
    np.testing.assert_array_equal(first['teacher_u'], fresh_data['teacher_u'])
    np.testing.assert_array_equal(first['teacher_log_w'], fresh_data['teacher_log_w'])
    assert probe.settings['proposal_components'] == 256
    assert probe.settings['proposals_per_component'] == 1
    assert probe.settings['teacher_sampling_mode'] == 'stratified'
    proposal_latents = jax.random.normal(proposal_keys[2], (2, 256, 2),
                                         dtype=probe.observations.dtype)
    proposal_mu, proposal_ls = conditional_sac.components(
        probe.state, probe.state.params, probe.observations, proposal_latents)
    reconstructed = ConditionalGaussianProposal(proposal_mu, proposal_ls, .05)
    expected_actions, expected_u, component_ids = reconstructed.sample(proposal_keys[3], 1, 'stratified')
    np.testing.assert_array_equal(component_ids, np.tile(np.arange(256), (2, 1)))
    np.testing.assert_array_equal(first['teacher_component_indices'], component_ids)
    np.testing.assert_array_equal(first['teacher_u'], expected_u)
    np.testing.assert_array_equal(first['teacher_actions'], expected_actions)
    np.testing.assert_allclose(first['teacher_log_q'], reconstructed.log_prob(expected_u)-2*np.log(40.),
                               rtol=1e-6, atol=1e-6)
    assert all(len(np.unique(row, axis=0)) == 256 for row in np.asarray(proposal_latents))
    probe.key = first['key']
    second = probe.prepare()
    assert np.array_equal(first['anchors'], second['anchors'])
    assert not np.array_equal(first['teacher_u'], second['teacher_u'])
    assert first['cost'].shape == (2, 4096, 16)
    assert first['teacher_u'].shape == (2, 256, 2)
    assert first['actor_z'].shape == (2, 16, 2)
    np.testing.assert_array_equal(first['pair_teacher_indices'], np.tile(np.arange(16), (2, 1)))
    np.testing.assert_allclose(first['ot_weights'], 1/16, rtol=0, atol=0)
    np.testing.assert_allclose(first['source_importance'], np.exp(first['source_log_importance']), rtol=1e-6)
    gradient = jax.grad(lambda actions: probe.q_fn(probe.observations, actions).sum())(first['teacher_actions'])
    assert np.isfinite(gradient).all() and np.any(np.asarray(gradient) != 0)
    np.testing.assert_allclose(first['teacher_q'], probe.target.log_prob(40*np.asarray(first['teacher_actions'])),
                               rtol=2e-5, atol=2e-5)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'actor.bin'
        agent.save(path)
        checkpoint_audit(path, 1)
        restored = GMM40V7(seed=seed, batch=2, latent_seed=seed, potential_solver=potential_solver)
        restored.restore(path)
        agent.advance(1)
        restored.advance(1)
        resume_error = tree_error(agent.checkpoint(), restored.checkpoint())
        assert resume_error == 0, resume_error
        wrong_bank = GMM40V7(seed=seed, batch=2, latent_seed=seed+1, potential_solver=potential_solver)
        try:
            wrong_bank.restore(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched fixed bank was accepted')
    return dict(status='passed', potential_solver=potential_solver,
                shared_public_update_state_max_error=state_error,
                shared_public_update_dual_max_error=dual_error,
                shared_public_update_rng_max_error=key_error,
                actor_initialization_and_teacher_identical_to_fresh_control=True,
                fresh_teacher_and_fixed_source_latents_verified=True,
                canonical_OT_and_actor_shapes_verified=True,
                teacher_component_ids_each_once=True,
                independent_proposal_latents=256,
                teacher_draws_per_component=1,
                teacher_actions_and_full_mixture_density_reconstructed=True,
                mismatched_fixed_bank_checkpoint_rejected=True,
                fixed_Q_action_gradient_finite_and_live=True,
                Q_callback_matches_original_GMM_log_density=True,
                checkpoint_resume_step1_to2_max_error=resume_error,
                scope='Separate two-update integration check; excluded from main smoke update count')


def make_agent(config):
    names = ('seed', 'batch', 'num_students', 'proposal_components', 'proposals_per_component',
             'teacher_sampling_mode', 'proposal_std', 'temperature', 'epsilon', 'iterations',
             'actor_samples', 'teacher_resample_count', 'latent_seed')
    return GMM40V7(**{name: config[name] for name in names}, hidden_dims=tuple(config['hidden_dims']),
                  potential_solver=config.get('potential_solver', 'fresh_sinkhorn'),
                  dual_learning_rate=config.get('dual_learning_rate', 1e-4),
                  dual_hidden_dims=tuple(config.get('dual_hidden_dims', (256, 256))),
                  actor_max_grad_norm=config.get('actor_max_grad_norm'),
                  source_importance_correction=config.get('source_importance_correction', True))


def source_diagnostics(agent):
    """Probe the next fresh map without consuming training RNG or updating params."""
    def probe(state, key, dual_state):
        data = agent.prepare_for(state, key, dual_state)
        log_mass = data['source_log_mass']
        mass = jnp.exp(log_mass)
        h = log_mass.shape[-1]
        log_weight = -jnp.log(float(h))-log_mass
        log_mean = (mass*log_weight).sum(-1)
        log_variance = (mass*(log_weight-log_mean[:, None])**2).sum(-1)
        weights = data['source_importance']
        teacher_weights = jnp.exp(data['teacher_log_w'])
        return dict(
            source_mass_tv_mean=(.5*jnp.abs(mass-1/h).sum(-1)).mean(),
            source_mass_tv_max=(.5*jnp.abs(mass-1/h).sum(-1)).max(),
            batch_mean_source_mass_tv=(.5*jnp.abs(mass.mean(0)-1/h).sum()),
            source_log_mass_min=log_mass.min(),
            source_log_importance_variance_under_source_mass=log_variance.mean(),
            source_log_importance_second_moment=jax.scipy.special.logsumexp(
                -2*jnp.log(float(h))-log_mass, axis=-1).mean(),
            sampled_log_importance_mean=data['source_log_importance'].mean(),
            sampled_log_importance_variance=data['source_log_importance'].var(),
            sampled_log_importance_max=data['source_log_importance'].max(),
            sampled_importance_mean=weights.mean(), sampled_importance_max=weights.max(),
            sampled_importance_ess_fraction=(weights.sum(-1)**2/(weights.shape[-1]*(weights**2).sum(-1))).mean(),
            teacher_ess_mean=(1/(teacher_weights**2).sum(-1)).mean(),
            row_max_absolute_error=jnp.abs(mass-1/h).max(),
            column_max_absolute_error=jnp.abs(data['ot']['teacher_mass']-data['ot_weights']).max())
    metrics = jax.jit(probe)(agent.state, agent.key, agent.dual_state)
    jax.block_until_ready(metrics)
    result = {name: float(value) for name, value in metrics.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--preflight', type=Path, help='Previously saved canonical CPU integration checks')
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=25)
    parser.add_argument('--potential-solver', choices=('persistent_dual', 'fresh_sinkhorn'), default='persistent_dual')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('This short-validation runner allows 1–1000 updates only')
    if args.chunk < 1 or args.iterations < 1 or (args.batch is not None and args.batch < 1):
        parser.error('Positive batch/chunk/iteration counts required')
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('Positive finite OT epsilon required')
    config = dict(profile='canonical_one_per_latent256_h4096_k16', seed=args.seed, steps=args.steps, chunk=args.chunk,
                  batch=args.batch, batch_smoke_override=args.batch != 256,
                  num_students=4096, proposal_components=256, proposals_per_component=1,
                  teacher_sampling_mode='stratified', actor_samples=16, teacher_resample_count=16,
                  teacher_component_allocation='one sample from each of 256 fresh independent latent conditionals',
                  latent_seed=args.seed,
                  potential_solver=args.potential_solver,
                  dual_learning_rate=1e-4, dual_hidden_dims=[256, 256],
                  proposal_std=.05, temperature=1., epsilon=args.epsilon, iterations=args.iterations,
                  hidden_dims=[256, 256], actor_learning_rate=3e-4,
                  actor_log_std_bounds=[-5., 1.], actor_initial_sigma=.5, latent_prior_std=1.,
                  action_scale=40., Q='log p_original(40 * normalized_action)',
                  scope='short fixed-Q implementation validation; not a convergence or MuJoCo result',
                  shared_update='optiq_dime.conditional_sac.update_actor',
                  actor_objective='mean[(b_i/m_i) * (T log pi_i(a|s) - Q(s,a) - T log r_i(a))]; no NLL or variance penalty',
                  source_importance_self_normalized=False, source_importance_clipped=False,
                  fixed_student_latents=True, fresh_teacher_every_update=True,
                  fresh_teacher_and_OT_every_update=args.potential_solver == 'fresh_sinkhorn',
                  auxiliary_dual_network=args.potential_solver == 'persistent_dual',
                  persistent_dual_state=args.potential_solver == 'persistent_dual',
                  evaluation_samples=10000, evaluation_seed=900000+args.seed,
                  evaluation_reference_seed=20260917,
                  cuda_visible=os.getenv('CUDA_VISIBLE_DEVICES'), pid=os.getpid())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'config.json', config)
    atomic_json(out/'status.json', dict(status='initializing', scope='short_validation', step=0, pid=os.getpid()))
    agent = None
    seconds = 0.
    try:
        (out/'checkpoints').mkdir()
        source = out/'source'
        for directory in ('gmm40', 'optiq_dime', 'common', 'models'):
            shutil.copytree(ROOT/directory, source/directory, ignore=shutil.ignore_patterns('__pycache__'))
        config['source_sha256'] = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in source.rglob('*') if path.is_file()}
        config['target_sha256'] = config['source_sha256']['gmm40/target_definition.json']
        atomic_json(out/'config.json', config)
        if args.preflight:
            checks = json.loads(args.preflight.read_text())
            assert checks['status'] == 'passed' and checks['seed'] == args.seed
            assert checks.get('potential_solver', 'fresh_sinkhorn') == args.potential_solver
            for relative, expected in checks['source_sha256'].items():
                assert config['source_sha256'][relative] == expected, relative
        else:
            with jax.default_device(jax.devices('cpu')[0]):
                checks = integration_checks(args.seed, args.potential_solver)
        atomic_json(out/'integration_checks.json', checks)
        agent = make_agent(config)
        reference = agent.target.sample(10000, config['evaluation_reference_seed'])
        full_reference = agent.target.sample(10000, config['evaluation_reference_seed'], bounded=False)
        goals = sorted(set([0, args.steps]+[step for step in (100, 200, 500, 1000) if step <= args.steps]))
        audits, warm_times, sizes_seen, probes = [], [], set(), []
        info = {}
        for goal in goals:
            while agent.updates < goal:
                count = min(args.chunk, goal-agent.updates)
                start = time.monotonic()
                info = agent.advance(count)
                elapsed = time.monotonic()-start
                seconds += elapsed
                assert int(agent.state.step) == agent.updates
                if agent.dual_state is not None:
                    assert int(agent.dual_state.step) == agent.updates
                if count in sizes_seen:
                    warm_times.append(elapsed/count)
                sizes_seen.add(count)
                if not all(np.isfinite(v) for v in info.values()):
                    raise FloatingPointError(info)
                assert info['actor_nll_used'] == 0
                assert info['ot_fresh_solve'] == float(args.potential_solver == 'fresh_sinkhorn')
                if args.potential_solver == 'persistent_dual':
                    assert info['ot_persistent_dual'] == info['ot_dual_updates'] == 1
                assert info['teacher_one_per_latent'] == 1
                assert info['teacher_proposal_component_count'] == 256
                row = dict(step=agent.updates, train_seconds=seconds, metrics=info)
                atomic_json(out/'status.json', dict(status='training', scope='short_validation', pid=os.getpid(), **row))
                with (out/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            path = out/'checkpoints'/f'step_{goal:07d}.bin'
            agent.save(path)
            audits.append(dict(step=goal, checkpoint=str(path), **checkpoint_audit(path, goal)))
            atomic_json(out/'checkpoint_audits.json', audits)
            evaluation_key = np.asarray(agent.key).copy()
            samples, extra = agent.evaluate_samples(10000, config['evaluation_seed'])
            assert np.array_equal(evaluation_key, np.asarray(agent.key))
            diagnostic = source_diagnostics(agent)
            probes.append(dict(step=goal, **diagnostic))
            atomic_json(out/'source_diagnostics.json', probes)
            result = save_evaluation(out, 'v7 one-per-latent256 validation', goal, samples, agent.target,
                                     reference, full_reference, dict(info, train_seconds=seconds), extra_samples=extra)
            print(json.dumps(dict(event='validation_evaluation', step=goal,
                                  near=result['high_density_fraction'], coverage=result['mode_coverage'],
                                  mmd2=result['mmd2'], training=info,
                                  source_diagnostics=diagnostic), allow_nan=False), flush=True)
        atomic_json(out/'runtime.json', dict(training_seconds=seconds, actual_updates=agent.updates,
                    median_warm_update_seconds=float(np.median(warm_times)) if warm_times else None,
                    timing_note='First block per chunk size excluded from warm timing; checkpoints/evaluation excluded'))
        atomic_json(out/'validation.json', dict(status='passed', scope='implementation smoke only',
                    actual_actor_updates=agent.updates, requested_actor_updates=args.steps,
                    integration_checks=checks, saved_checkpoint_counts_verified=True,
                    final_metrics=result, final_training_metrics=info, source_diagnostics=probes,
                    convergence_claimed=False, source_marginals_exact_claimed=False,
                    ot_note='Source marginal errors are reported; importance weights are not clipped or self-normalized'))
        atomic_json(out/'status.json', dict(status='completed', scope='short_validation', step=agent.updates,
                    actor_updates=int(agent.state.step), pid=os.getpid(), train_seconds=seconds))
    except Exception as error:
        failure_checkpoint = None
        if agent is not None:
            failure_checkpoint = out/'checkpoints'/f'failure_step_{agent.updates:07d}.bin'
            agent.save(failure_checkpoint)
        atomic_json(out/'status.json', dict(status='failed', scope='short_validation', error=repr(error),
                    step=agent.updates if agent else 0, pid=os.getpid(),
                    failure_checkpoint=str(failure_checkpoint) if failure_checkpoint else None))
        raise


if __name__ == '__main__':
    main()
