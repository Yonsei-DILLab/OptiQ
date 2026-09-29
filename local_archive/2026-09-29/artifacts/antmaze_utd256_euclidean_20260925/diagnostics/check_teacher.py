"""Read-only current-branch teacher probe at the original fixed maze start."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('v3', 'v4'), required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--continuations', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    import flax.serialization as fs
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import torch
    from omegaconf import OmegaConf
    from antmaze_experiments.envs import vector, transition

    run = args.root / 'runs' / f'{args.task}-optiq-utd1-basic_euclidean-s0'
    config = json.loads((run / 'config.json').read_text())
    checkpoint = run / 'policy-checkpoints' / f'step_{args.step:010d}' / 'policy.pt'
    proof = json.loads((checkpoint.parent / 'verification.json').read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == proof['sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['config'] == config

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    spec = importlib.util.spec_from_file_location(
        'frozen_trg', args.source / 'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    native = OmegaConf.create(config['native'])
    native.output_root = '/tmp/optiq-antmaze-teacher-readonly'
    model = module.runner.OptiQDIME(
        'MlpPolicy', env=Descriptor(), cfg=native,
        model_save_path=None, save_every_n_steps=config['steps'])
    policy = model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state)
    saved = fs.from_bytes(template, payload['learner']['policy'])
    policy.actor_state = saved['actor']
    policy.qf_state = saved['critic']
    policy.target_actor_state = saved['target_actor']

    env = vector(args.task, 1, seed=87231, asynchronous=False, fixed=True,
                 reward_profile=config['reward_profile'], random_init=False)
    try:
        obs = np.asarray(env.reset()[:1], dtype=np.float32)
    finally:
        env.close()
    actor = native.alg.actor
    count = int(actor.num_policy_samples)
    assert count == 64 and int(actor.proposals_per_policy_sample) == 1
    assert actor.distillation_loss == 'direct_gmm_nll'
    assert actor.source_q_eval == 'mean'
    key, latent_key, proposal_key, _ = jax.random.split(jax.random.PRNGKey(950001), 4)
    z_key, _ = jax.random.split(latent_key)
    z = jax.random.normal(z_key, (count, 8), dtype=jnp.float32)
    mu, log_std = policy.actor_state.apply_fn(
        {'params': policy.actor_state.params},
        jnp.asarray(np.repeat(obs, count, axis=0)), z)
    mu, log_std = mu[None], log_std[None]
    proposal = ConditionalGaussianProposal(mu, log_std, float(actor.proposal_std))
    sampled_actions, u, _ = proposal.sample(proposal_key, 1, 'exact')
    actions = np.asarray(sampled_actions).reshape(count, 8)
    log_q = np.asarray(proposal.log_prob(u)).reshape(count)
    predicted = policy.qf_state.apply_fn(
        {'params': policy.qf_state.params,
         'batch_stats': policy.qf_state.batch_stats},
        jnp.asarray(np.repeat(obs, count, axis=0)), jnp.asarray(actions),
        train=False)
    q_twins = np.asarray(predicted)[..., 0]
    assert q_twins.shape == (2, count)
    q = q_twins.mean(axis=0)
    temperature = float(actor.temperature)
    beta = float(actor.density_beta)
    q_weights = np.asarray(jax.nn.softmax(jnp.asarray(q / temperature)))
    weights = np.asarray(jax.nn.softmax(jnp.asarray(q / temperature - beta * log_q)))
    result = dict(task=args.task, step=args.step, source_commit=config['source_commit'],
                  checkpoint_sha256=digest, read_only=True, candidates=count,
                  temperature=temperature, density_beta=beta,
                  source_q_std=float(q.std()), q_logit_std=float(q.std() / temperature),
                  density_logit_std=float(beta * log_q.std()),
                  source_q_min=float(q.min()), source_q_max=float(q.max()),
                  q_only_ess=float(1 / np.square(q_weights).sum()),
                  full_teacher_ess=float(1 / np.square(weights).sum()),
                  full_max_weight=float(weights.max()),
                  mu_coordinate_sd=float(np.asarray(mu).reshape(count, 8).std(axis=0).mean()),
                  conditional_sigma_mean=float(np.exp(np.asarray(log_std)).mean()),
                  caveat='One initial state and one 64-candidate cloud; actions are not route-labeled.')
    if args.continuations:
        assert 0 < args.continuations <= 4
        masses = []
        for rep in range(args.continuations):
            env = vector(args.task, count, seed=87231, asynchronous=False, fixed=True,
                         reward_profile=config['reward_profile'], random_init=False)
            try:
                observations = np.asarray(env.reset(), dtype=np.float32)
                common = env.envs[0].initial
                for i, single in enumerate(env.envs):
                    single.initial = common
                    observations[i] = single.restore(common)
                np.testing.assert_allclose(observations, np.repeat(obs, count, axis=0))
                paths = [[o[:2].copy()] for o in observations]
                active = np.ones(count, bool)
                for t in range(700):
                    if t == 0:
                        chosen = actions
                    else:
                        chosen = np.asarray(policy.sample_action(
                            policy.actor_state, jnp.asarray(observations),
                            jax.random.PRNGKey(951000 + rep * 1000 + t),
                            deterministic=False, sample_conditional_noise=False))
                    nxt, _, done, infos = env.step(chosen)
                    final, _ = transition(nxt, done, infos)
                    for i in np.flatnonzero(active):
                        paths[i].append(final[i, :2].copy())
                        if done[i]:
                            active[i] = False
                    observations = nxt
                    if not active.any():
                        break
            finally:
                env.close()
            labels = []
            for path in paths:
                xy = np.asarray(path)
                if args.task == 'v3':
                    left = bool(np.any(xy[:, 0] < -8))
                    right = bool(np.any(xy[:, 0] > 8))
                    label = ('both' if left and right else 'left' if left else
                             'right' if right else 'uncommitted')
                else:
                    crossings = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
                    if len(crossings):
                        i = int(crossings[0])
                        fraction = (-4 - xy[i, 0]) / (xy[i + 1, 0] - xy[i, 0])
                        y = xy[i, 1] + fraction * (xy[i + 1, 1] - xy[i, 1])
                        label = 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted'
                    else:
                        label = 'uncommitted'
                labels.append(label)
            masses.append({label: dict(count=labels.count(label),
                                       teacher_mass=float(np.sum(weights[np.asarray(labels) == label])))
                           for label in sorted(set(labels))})
        result['forced_first_teacher_candidate_continuation'] = dict(
            candidate_action='one full-policy proposal per latent component',
            subsequent_policy='random-z mu-only',
            initial_full_state='identical across all 64 candidates',
            replicates=masses,
            limitation='A single first action does not determine a long route; continuation randomness and later policy actions matter.')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
