"""Read-only critic diagnostics using existing evaluation trajectories.

No optimizer/update calls or training RNG draws. Q/MC comparisons use the current
full policy; path labels are retrospective and are not causal action-mode labels.
"""
import json
from pathlib import Path

import numpy as np


LANDMARKS = (0, 10, 25, 50, 100, 200)


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _stats(values):
    a = np.asarray(values, np.float64).ravel()
    if not len(a):
        return dict(n=0, mean=None, sd=None, mean_abs=None, rms=None)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(n=len(a), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if len(a) > 1 else 0.,
                mean_abs=float(np.abs(a).mean()), rms=float(np.sqrt(np.square(a).mean())))


def route_label(task, xy):
    """Return observed gate label and transition index immediately before entry."""
    xy = np.asarray(xy)
    if task in ('v2', 'v3'):
        threshold = 4 if task == 'v2' else 8
        left = np.flatnonzero(xy[:, 0] < -threshold)
        right = np.flatnonzero(xy[:, 0] > threshold)
        label = 'both' if len(left) and len(right) else 'left' if len(left) else 'right' if len(right) else 'uncommitted'
        entered = np.r_[left, right]
        return label, max(0, int(entered.min()) - 1) if len(entered) else -1
    if task not in ('v1', 'v4'):
        raise ValueError(task)
    indices = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
    if not len(indices):
        return 'uncommitted', -1
    i = int(indices[0])
    fraction = (-4 - xy[i, 0]) / (xy[i + 1, 0] - xy[i, 0])
    y = xy[i, 1] + fraction * (xy[i + 1, 1] - xy[i, 1])
    return 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted', i


def decompose(rewards, terminals, online_mean, next_online_mean,
              next_target_mean, next_target_min, gamma):
    """Exact algebraic split, not proof of a causal effect of changing tau."""
    r, q, no, nt, nm = [np.asarray(x, np.float64) for x in
                         (rewards, online_mean, next_online_mean, next_target_mean, next_target_min)]
    discount = gamma * (1 - np.asarray(terminals, np.float64))
    fit = r + discount * nm - q
    lag = discount * (no - nt)
    twin = discount * (nt - nm)
    total = r + discount * no - q
    np.testing.assert_allclose(fit + lag + twin, total, rtol=1e-10, atol=1e-9)
    return dict(td_fit_residual=fit, target_tracking_term=lag,
                target_twin_min_term=twin, online_bellman_residual=total)


def discounted_returns(rewards, gamma, final_value=0.):
    result = np.empty(len(rewards), np.float64)
    running = float(final_value)
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        result[i] = running
    return result


def _q_function():
    # Lazy import keeps pure diagnostic math testable without JAX/MuJoCo.
    import jax
    @jax.jit
    def call(state, observations, actions):
        online = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats},
                                observations, actions, train=False)[..., 0].T
        target = state.apply_fn({'params': state.target_params, 'batch_stats': state.target_batch_stats},
                                observations, actions, train=False)[..., 0].T
        return online, target
    return call


_Q_CALL = None


def evaluate_q(policy, observations, actions, batch_size=1024):
    global _Q_CALL
    import jax.numpy as jnp
    if _Q_CALL is None:
        _Q_CALL = _q_function()
    observations, actions = np.asarray(observations), np.asarray(actions)
    if len(observations) != len(actions) or not len(actions):
        raise ValueError('Nonempty aligned observations/actions required')
    online, target = [], []
    for start in range(0, len(actions), batch_size):
        o, a = observations[start:start + batch_size], actions[start:start + batch_size]
        n = len(a)
        o = np.pad(o, ((0, batch_size - n), (0, 0)), mode='edge')
        a = np.pad(a, ((0, batch_size - n), (0, 0)), mode='edge')
        q, qt = _Q_CALL(policy.qf_state, jnp.asarray(o), jnp.asarray(a))
        online.append(np.asarray(q)[:n]); target.append(np.asarray(qt)[:n])
    online, target = np.concatenate(online), np.concatenate(target)
    if online.shape != (len(actions), 2) or target.shape != online.shape:
        raise ValueError('Diagnostics require scalar twin critics')
    if not np.isfinite(online).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite critic output')
    return online, target


def policy_values(policy, observations, seed, draws=64):
    """Independent diagnostic key; neither reset_noise nor mutable training RNG."""
    import jax
    import jax.numpy as jnp
    obs = np.repeat(np.asarray(observations), draws, axis=0)
    actions = np.asarray(policy.sample_action(policy.actor_state, jnp.asarray(obs),
        jax.random.PRNGKey(seed), deterministic=False, sample_conditional_noise=True))
    q, qt = evaluate_q(policy, obs, actions)
    shape = (len(observations), draws)
    return dict(online_mean=q.mean(-1).reshape(shape).mean(-1),
                target_mean=qt.mean(-1).reshape(shape).mean(-1),
                target_min=qt.min(-1).reshape(shape).mean(-1))


def teacher_probe(learner, observations, seed, reward_multiplier=1.):
    """Recompute a bounded current teacher cloud, with no route assignment.

    The finite actor mixture is constructed BEFORE its candidates are sampled;
    the very same mixture supplies log q, matching the training implementation.
    """
    import jax
    import jax.numpy as jnp
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    p = learner.model.policy
    cfg = learner.model.cfg.alg.actor
    assert cfg.type == 'semi_implicit' and cfg.distillation_loss == 'direct_gmm_nll'
    assert cfg.teacher_distribution == 'conditional_mixture'
    assert cfg.get('latent_prior', 'normal') == 'normal'
    assert cfg.num_policy_samples == 64 and cfg.proposals_per_policy_sample == 1
    assert cfg.proposal_sampling_mode == 'exact' and not cfg.include_anchor
    assert cfg.density_correction and not cfg.adaptive_density_beta
    assert cfg.source_q_eval == 'mean' and cfg.get('soft_proximal_ess_fraction', 0.) == 0.
    assert not (cfg.get('temperature_schedule') or {}).get('enabled', False)
    observations = np.asarray(observations, np.float32)
    batch, components = len(observations), int(cfg.num_policy_samples)
    dim = int(p.actor_state.params['mu']['bias'].shape[0])
    _, latent_key, proposal_key, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
    z_key, _ = jax.random.split(latent_key)
    latents = jax.random.normal(z_key, (batch, components, dim), dtype=jnp.float32)
    repeated = np.repeat(observations, components, axis=0)
    mu, log_std = p.actor_state.apply_fn({'params': p.actor_state.params},
        jnp.asarray(repeated), latents.reshape(batch * components, dim))
    mu, log_std = mu.reshape(batch, components, dim), log_std.reshape(batch, components, dim)
    proposal = ConditionalGaussianProposal(mu, log_std, float(cfg.proposal_std))
    candidates, actions, component_indices = proposal.sample(proposal_key, 1, 'exact')
    logq = np.asarray(proposal.log_prob(actions))
    q, qt = evaluate_q(p, repeated, np.asarray(candidates).reshape(batch * components, dim))
    q, qt = q.reshape(batch, components, 2), qt.reshape(batch, components, 2)
    qmean = q.mean(-1)
    logits = qmean / float(cfg.temperature) - float(cfg.density_beta) * logq
    weights = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
    q_only = np.asarray(jax.nn.softmax(jnp.asarray(qmean / float(cfg.temperature)), axis=-1))
    arrays = dict(observations=observations, latents=np.asarray(latents),
        mu=np.asarray(mu), log_std=np.asarray(log_std), candidates=np.asarray(candidates),
        component_indices=np.asarray(component_indices), log_proposal_density=logq,
        q_online=q, q_target=qt, weights=weights, q_only_weights=q_only)
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite teacher diagnostic')
    np.testing.assert_allclose(weights.sum(-1), 1., rtol=1e-6, atol=1e-6)
    q_spread = qmean.std(-1)
    gain = (weights * qmean).sum(-1) - qmean.mean(-1)
    summary = dict(seed=int(seed), states=batch, components=components, candidates_per_state=components,
        temperature=float(cfg.temperature), density_beta=float(cfg.density_beta),
        proposal_std_floor=float(cfg.proposal_std), source_q='online twin mean',
        source_ess=_stats(1. / np.square(weights).sum(-1)),
        q_only_ess=_stats(1. / np.square(q_only).sum(-1)),
        max_source_weight=_stats(weights.max(-1)),
        q_std=dict(raw=_stats(q_spread), normalized=_stats(q_spread / reward_multiplier)),
        q_logit_std=_stats(q_spread / float(cfg.temperature)),
        density_logit_std=_stats(float(cfg.density_beta) * logq.std(-1)),
        teacher_q_gain=dict(raw=_stats(gain), normalized=_stats(gain / reward_multiplier)),
        limitation='Candidate actions are not assigned trajectory routes. Reference route labels describe the recorded episode only; no forced-action continuation rollout is added.')
    return arrays, summary


def _guard(learner):
    p, m = learner.model.policy, learner.model
    import jax
    def key_bytes(key):
        return None if key is None else np.asarray(jax.random.key_data(key)).tobytes()
    return (id(p.actor_state), id(p.qf_state), id(p.target_actor_state),
            key_bytes(p.key), key_bytes(p.noise_key), key_bytes(m.key), int(m._n_updates))


class EvaluationTraceRecorder:
    """Hook before the evaluator changes its active mask; only records live rows."""
    def __init__(self):
        self.episodes = []
        self.batch = None

    def start_batch(self, active):
        if self.batch is not None:
            raise RuntimeError('Previous evaluation batch is still open')
        self.batch = []
        for included in active:
            row = {k: [] for k in ('observations', 'actions', 'rewards', 'next_observations', 'dones', 'terminals')} if included else None
            self.batch.append(row)
            if row is not None:
                self.episodes.append(row)

    def record(self, obs, actions, rewards, final_obs, dones, terminals, active):
        if self.batch is None:
            raise RuntimeError('start_batch must be called first')
        for i in np.flatnonzero(active):
            row = self.batch[i]
            if row is None or (row['dones'] and row['dones'][-1]):
                raise ValueError('Attempt to record an excluded/completed episode')
            for name, values in zip(row, (obs, actions, rewards, final_obs, dones, terminals)):
                row[name].append(np.asarray(values[i]).copy())

    def end_batch(self):
        if self.batch is None:
            raise RuntimeError('No open evaluation batch')
        for row in self.batch:
            if row is not None and (not row['dones'] or not row['dones'][-1]):
                raise ValueError('Diagnostic episode did not reach termination/time limit')
        self.batch = None

    def save(self, learner, task, destination, step, mode='policy', reward_multiplier=1., goals=None):
        if self.batch is not None or not self.episodes:
            raise ValueError('Finish recorded episodes before saving')
        if learner.method != 'optiq' or mode != 'policy':
            raise ValueError('Q/MC calibration requires OptiQ direct full-policy evaluation')
        if not np.isfinite(reward_multiplier) or reward_multiplier <= 0:
            raise ValueError('Positive reward multiplier required')
        before = _guard(learner)
        p, gamma = learner.model.policy, float(learner.model.gamma)
        assert learner.model.cfg.alg.critic.n_atoms == 1
        rows = [{k: np.asarray(v) for k, v in row.items()} for row in self.episodes]
        lengths = np.array([len(r['rewards']) for r in rows], np.int64)
        offsets = np.r_[0, np.cumsum(lengths)]
        bank = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        q, qt = evaluate_q(p, bank['observations'], bank['actions'])
        boundary = policy_values(p, np.stack([r['next_observations'][-1] for r in rows]),
                                 seed=(1900003 + int(step)) % (2 ** 32), draws=64)
        labels, gates, decompositions, raw_returns, boot_returns = [], [], [], [], []
        episode_sums = []
        for i, row in enumerate(rows):
            start, end = offsets[i:i + 2]
            if len(row['rewards']) > 1:
                np.testing.assert_array_equal(row['next_observations'][:-1], row['observations'][1:])
            if np.any(row['dones'][:-1]) or np.any(row['terminals'][:-1]):
                raise ValueError('Episode contains transitions after completion')
            label, gate = route_label(task, np.r_[row['observations'][:1, :2], row['next_observations'][:, :2]])
            labels.append(label); gates.append(gate)
            no = np.r_[q[start + 1:end].mean(-1), boundary['online_mean'][i]]
            nt = np.r_[qt[start + 1:end].mean(-1), boundary['target_mean'][i]]
            nm = np.r_[qt[start + 1:end].min(-1), boundary['target_min'][i]]
            terms = decompose(row['rewards'], row['terminals'], q[start:end].mean(-1), no, nt, nm, gamma)
            final_v = 0. if row['terminals'][-1] else boundary['online_mean'][i]
            raw = discounted_returns(row['rewards'], gamma)
            boot = discounted_returns(row['rewards'], gamma, final_v)
            sums = {k: float(np.dot(gamma ** np.arange(len(v)), v)) for k, v in terms.items()}
            np.testing.assert_allclose(sums['online_bellman_residual'], boot[0] - q[start].mean(), rtol=1e-6, atol=1e-5)
            sums.update(raw_mc_minus_q0=float(raw[0] - q[start].mean()),
                        bootstrap_mc_minus_q0=float(boot[0] - q[start].mean()))
            episode_sums.append(sums); decompositions.append(terms)
            raw_returns.append(raw); boot_returns.append(boot)
        bank.update(q_online=q, q_target=qt, offsets=offsets, routes=np.asarray(labels),
                    gate_transition=np.asarray(gates), lengths=lengths,
                    discounted_return=np.concatenate(raw_returns),
                    bootstrapped_discounted_return=np.concatenate(boot_returns))
        bank.update({k: np.concatenate([d[k] for d in decompositions]) for k in decompositions[0]})
        bank.update({'boundary_' + k: v for k, v in boundary.items()})
        if goals is not None:
            assert len(goals) == len(rows)
            bank['goals'] = np.asarray(goals)
        summary = dict(step=int(step), task=task, mode=mode, gamma=gamma,
            reward_multiplier=float(reward_multiplier), episodes=len(rows),
            online_Q='current twin mean; actor teacher uses this',
            TD_target='current full policy action; minimum of EMA target twins',
            units='Raw arrays use training reward units; normalized summaries divide by reward_multiplier.',
            limits=['Routes are retrospective whole-trajectory gate labels, not causal action labels.',
                'At t>0 different routes generally visit different full states.',
                'Raw MC stops at timeout; bootstrap MC uses the same online critic, not independent truth.',
                'Residual decomposition is algebraic and does not establish an intervention effect.'],
            routes={})
        for label in sorted(set(labels)):
            ids = [i for i, value in enumerate(labels) if value == label]
            entries = {}
            for mark in [*LANDMARKS, 'first_gate']:
                ix = [offsets[i] + (gates[i] if mark == 'first_gate' else mark) for i in ids
                      if (gates[i] >= 0 if mark == 'first_gate' else lengths[i] > mark)]
                if not ix:
                    continue
                metrics = dict(q_online=q[ix].mean(-1), q_target=qt[ix].mean(-1),
                    target_twin_gap=qt[ix, 0] - qt[ix, 1],
                    raw_mc=bank['discounted_return'][ix], bootstrap_mc=bank['bootstrapped_discounted_return'][ix],
                    raw_mc_minus_q=bank['discounted_return'][ix] - q[ix].mean(-1))
                metrics.update({k: bank[k][ix] for k in decompositions[0]})
                entries[str(mark)] = {k: dict(raw=_stats(v), normalized=_stats(np.asarray(v) / reward_multiplier)) for k, v in metrics.items()}
            summary['routes'][label] = dict(episodes=len(ids), landmarks=entries,
                initial_discounted_decomposition={k: dict(raw=_stats([episode_sums[i][k] for i in ids]),
                    normalized=_stats([episode_sums[i][k] / reward_multiplier for i in ids])) for k in episode_sums[0]})
        reference_indices, reference_episodes, reference_marks = [], [], []
        for i, length in enumerate(lengths):
            for mark in (0, 25, 50, 'first_gate'):
                t = gates[i] if mark == 'first_gate' else mark
                if 0 <= t < length:
                    reference_indices.append(int(offsets[i] + t))
                    reference_episodes.append(i)
                    reference_marks.append(str(mark))
        teacher_arrays, summary['teacher_probe'] = teacher_probe(learner,
            bank['observations'][reference_indices], seed=(2300009 + int(step)) % (2 ** 32),
            reward_multiplier=reward_multiplier)
        teacher_arrays.update(reference_transition_indices=np.asarray(reference_indices),
            reference_episode_indices=np.asarray(reference_episodes),
            reference_landmarks=np.asarray(reference_marks),
            reference_episode_routes=np.asarray(labels)[reference_episodes])
        if _guard(learner) != before:
            raise AssertionError('Inference diagnostics changed learner state or RNG')
        summary['verification'] = dict(training_states_and_rng_unchanged=True,
            no_extra_rollout=True, residual_identity_verified=True, boundary_policy_draws=64)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / 'critic-trace.npz', **bank)
        np.savez_compressed(destination / 'teacher-probe.npz', **teacher_arrays)
        _write(destination / 'critic-diagnostics.json', summary)
        return summary


def replay_diagnostics(learner, step, folder, reward_multiplier=1., samples=1024, draws=8):
    """Uniform deterministic private replay sample; does not call sample_batch."""
    if learner.method != 'optiq' or learner.noveld_enabled:
        raise ValueError('Replay diagnostics currently require OptiQ and NovelD OFF')
    if not np.isfinite(reward_multiplier) or reward_multiplier <= 0:
        raise ValueError('Positive reward multiplier required')
    before = _guard(learner)
    memory, p = learner.replay, learner.model.policy
    count = min(samples, int(memory.cur_capacity))
    if count <= 0:
        raise ValueError('Empty replay')
    indices = np.random.default_rng(910019 + int(step)).choice(int(memory.cur_capacity), count, replace=False)
    import torch
    ix = torch.as_tensor(indices, device=memory.buf_obs.device, dtype=torch.long)
    arrays = {k: getattr(memory, 'buf_' + k)[ix].detach().cpu().numpy()
              for k in ('obs', 'action', 'reward', 'next_obs', 'done')}
    q, qt = evaluate_q(p, arrays['obs'], arrays['action'])
    values = policy_values(p, arrays['next_obs'], seed=(1700021 + int(step)) % (2 ** 32), draws=draws)
    terms = decompose(arrays['reward'].ravel(), arrays['done'].ravel(), q.mean(-1),
                      values['online_mean'], values['target_mean'], values['target_min'], float(learner.model.gamma))
    metrics = dict(q_online=q.mean(-1), q_target=qt.mean(-1), online_minus_target=q.mean(-1)-qt.mean(-1),
                   reward=arrays['reward'].ravel(), **terms)
    xy = arrays['obs'][:, :2]
    regions = dict(left_of_x_minus8=xy[:, 0] < -8, right_of_x_plus8=xy[:, 0] > 8,
                   upper_after_x_minus4=(xy[:, 0] < -4) & (xy[:, 1] > 2),
                   lower_after_x_minus4=(xy[:, 0] < -4) & (xy[:, 1] < -2))
    spatial = {name: int(mask.sum()) for name, mask in regions.items()}
    result = dict(step=int(step), samples=count, replay_capacity=int(memory.cur_capacity),
        reward_multiplier=float(reward_multiplier), policy_draws=draws, spatial_counts=spatial,
        spatial_note='Coordinate-region counts only; these are not complete trajectory route labels.',
        metrics={k: dict(raw=_stats(v), normalized=_stats(v / reward_multiplier)) for k, v in metrics.items()},
        spatial_metrics={name: {k: dict(raw=_stats(v[mask]),
            normalized=_stats(v[mask] / reward_multiplier)) for k, v in metrics.items()}
            for name, mask in regions.items()})
    if _guard(learner) != before:
        raise AssertionError('Replay diagnostics changed learner state or RNG')
    destination = Path(folder) / 'replay-diagnostics' / f'{int(step):010d}'
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'sample.npz', indices=indices, **arrays, q_online=q, q_target=qt,
                        **{'next_' + k: v for k, v in values.items()}, **terms)
    result['training_states_and_rng_unchanged'] = True
    _write(destination / 'summary.json', result)
    return result
