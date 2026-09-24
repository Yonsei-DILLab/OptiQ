"""Paired CPU inference: native episode limit versus extra diagnostic time.

The checkpoint, policy sampler, goal radius, reward and training remain frozen.
Extended-horizon successes must never be reported as native-budget successes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--extended-limit', type=int, default=1400)
    parser.add_argument('--cpu-offset', type=int, default=48)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 1000 and 700 < args.extended_limit <= 2100
    assert not os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('JAX_PLATFORMS') == 'cpu'
    report_source = Path(__file__).resolve().parents[1]
    report_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=report_source, text=True).strip()
    cfg = json.loads((args.run / 'config.json').read_text())
    assert cfg['method'] == 'optiq' and cfg['task'] in ('v3', 'v4')
    assert cfg['effective_eval_starts'] == 'fixed'
    source = Path('/home/heechan/OptiQ-ops/sources') / cfg['source_commit']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == cfg['source_commit']
    assert args.checkpoint.is_relative_to(args.run)
    proof_path = (args.checkpoint.parent / 'verification.json' if args.checkpoint.name == 'policy.pt'
                  else args.run / 'checkpoint-verification.json')
    proof = json.loads(proof_path.read_text())
    assert proof['readback_verified'] and sha(args.checkpoint) == proof['sha256']
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [cpus[(args.cpu_offset + i) % len(cpus)] for i in range(min(4, len(cpus)))])
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import jax
    import flax.serialization as fs
    import gymnasium as gym
    from omegaconf import OmegaConf
    from antmaze_experiments.learners import JaxLearner, evaluation_rng
    from antmaze_experiments.envs import vector, transition
    from antmaze_experiments.progress_reward import PROFILES, distance, progress_scale, step_cost, bonus_enabled
    from antmaze_experiments.critic_diagnostics import route_label
    torch.set_num_threads(1)
    assert all(device.platform == 'cpu' for device in jax.devices())
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert payload['config'] == cfg
    step = int(payload['step'])
    assert step == (proof['step'] if 'step' in proof else proof['steps'])

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
        def reset(self, **kwargs):
            raise RuntimeError('Inference descriptor only')
        def step(self, action):
            raise RuntimeError('Inference descriptor only')

    spec = importlib.util.spec_from_file_location('frozen_trg_train', source / 'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = OmegaConf.create(cfg['native'])
    native.output_root = str(args.output / 'constructor')
    model = module.runner.OptiQDIME('MlpPolicy', env=Descriptor(), cfg=native,
                                  model_save_path=None, save_every_n_steps=cfg['steps'])
    policy = model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state, target_actor=policy.target_actor_state)
    saved = fs.from_bytes(template, payload['learner']['policy'])
    original = fs.msgpack_restore(payload['learner']['policy'])

    def equal(actual, wanted):
        if isinstance(wanted, dict):
            assert set(actual) == set(wanted)
            for key in wanted:
                equal(actual[key], wanted[key])
        elif isinstance(wanted, (list, tuple)):
            assert len(actual) == len(wanted)
            for a, b in zip(actual, wanted):
                equal(a, b)
        else:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(wanted))

    equal(fs.to_state_dict(saved), original)
    policy.actor_state = saved['actor']
    policy.qf_state = saved['critic']
    policy.target_actor_state = saved['target_actor']
    policy.key = jax.random.PRNGKey(700000 + step)
    policy.noise_key = jax.random.PRNGKey(700000 + step)

    def state_digest():
        states = dict(actor=policy.actor_state, critic=policy.qf_state, target_actor=policy.target_actor_state)
        return hashlib.sha256(fs.to_bytes(states)).hexdigest()

    before = state_digest()
    before_rng = np.asarray(jax.random.key_data(policy.key)).copy(), np.asarray(jax.random.key_data(policy.noise_key)).copy()
    learner = JaxLearner.__new__(JaxLearner)
    learner.method, learner.model, learner.reward_profile = 'optiq', model, cfg['reward_profile']
    provenance = dict(training_source=cfg['source_commit'], evaluation_source=report_sha,
        parent_run=str(args.run), checkpoint=str(args.checkpoint), checkpoint_sha256=proof['sha256'],
        checkpoint_step=step, checkpoint_updates=int(payload['updates']), training_seed=cfg['seed'], task=cfg['task'],
        episodes_per_condition=args.episodes, native_limit=700, extended_limit=args.extended_limit,
        reset='original identical full state at origin', mode='policy', conditional_sigma=True,
        random_latent=True, external_dacer_noise=False, intrinsic_reward=False, inference_only=True,
        primary_evaluation_unchanged=True, goal_radius_unchanged=True,
        paired_rng='700000+checkpoint_step+100003*batch; restored around each condition',
        limitation='Extended-limit success is supplementary; it does not satisfy the native 700-step benchmark.',
        training_config=cfg)
    write(args.output / 'provenance.json', provenance)
    started = time.monotonic()
    all_results = {}
    count = min(20, args.episodes)
    for limit in (700, args.extended_limit):
        records = []
        env = vector(cfg['task'], count, seed=87231, asynchronous=False, fixed=True,
                     reward_profile=cfg['reward_profile'], random_init=False)
        try:
            for one in env.envs:
                assert one.env._max_episode_steps == 700
                one.env._max_episode_steps = limit
            for batch in range((args.episodes + count - 1) // count):
                with evaluation_rng(learner, 700000 + step + 100003 * batch):
                    obs = env.reset()
                    common = env.envs[0].initial
                    for i, one in enumerate(env.envs):
                        one.initial = common
                        obs[i] = one.restore(common)
                    initial = [np.r_[one.state()['qpos'], one.state()['qvel']] for one in env.envs]
                    active = np.arange(count) + batch * count < args.episodes
                    tracks = [[o[:2].copy()] for o in obs]
                    rewards, goals, lengths = np.zeros(count), np.zeros(count, int), np.zeros(count, int)
                    for _ in range(limit):
                        actions = learner.act(obs, 'policy')
                        nxt, r, done, infos = env.step(actions)
                        final, _ = transition(nxt, done, infos)
                        for i in np.flatnonzero(active):
                            tracks[i].append(final[i, :2].copy())
                            rewards[i] += r[i]
                            lengths[i] += 1
                            if done[i]:
                                goals[i] = int(infos[i].get('success', 0))
                                active[i] = False
                        obs = nxt
                        if not active.any():
                            break
                    for i in range(min(count, args.episodes - batch * count)):
                        records.append(dict(xy=np.asarray(tracks[i]), returns=float(rewards[i]),
                            goal=int(goals[i]), length=int(lengths[i]), initial=initial[i]))
                write(args.output / 'progress.json', dict(limit=limit, completed_episodes=len(records),
                    episodes_per_condition=args.episodes, seconds=time.monotonic() - started))
        finally:
            env.close()
        all_results[limit] = records

    max_reward_error = 0.
    summaries = {}
    for limit, records in all_results.items():
        xy = np.full((args.episodes, max(x['length'] for x in records) + 1, 2), np.nan, np.float32)
        route_counts, successful_routes = {}, {}
        for i, item in enumerate(records):
            path = item['xy']
            assert np.isfinite(path).all() and np.isfinite(item['returns'])
            xy[i, :len(path)] = path
            route, _ = route_label(cfg['task'], path)
            route_counts[route] = route_counts.get(route, 0) + 1
            if item['goal']:
                successful_routes[route] = successful_routes.get(route, 0) + 1
            profile = cfg['reward_profile']
            if profile in PROFILES:
                assert not bonus_enabled(profile), 'This diagnostic targets no-bonus progress profiles'
                expected = progress_scale(profile) * (distance(path[0], cfg['task'], profile) - distance(path[-1], cfg['task'], profile)) - step_cost(profile) * item['length']
                max_reward_error = max(max_reward_error, abs(item['returns'] - float(expected)))
        initial = np.asarray([x['initial'] for x in records])
        np.testing.assert_array_equal(initial, np.repeat(initial[:1], args.episodes, axis=0))
        np.testing.assert_array_equal(initial[:, :2], np.zeros((args.episodes, 2)))
        raw = args.output / f'rollouts-limit{limit}.npz'
        np.savez_compressed(raw, xy=xy, returns=[x['returns'] for x in records],
            lengths=[x['length'] for x in records], goals=[x['goal'] for x in records],
            initial_full_state=initial, env_steps=np.array(step), mode='policy', fixed=True, episode_limit=limit)
        summaries[str(limit)] = dict(episodes=args.episodes, routes=route_counts, successful_routes=successful_routes,
            successes=sum(bool(x['goal']) for x in records), mean_length=float(np.mean([x['length'] for x in records])),
            raw_path=raw.name, raw_sha256=sha(raw))
    for original, extended in zip(all_results[700], all_results[args.extended_limit]):
        n = original['length']
        assert extended['length'] >= n
        np.testing.assert_array_equal(original['initial'], extended['initial'])
        np.testing.assert_array_equal(original['xy'], extended['xy'][:n + 1])
        if original['goal']:
            assert extended['goal'] == original['goal'] and extended['length'] == n
        else:
            assert n == 700
    assert max_reward_error < .01 and state_digest() == before
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(policy.key)), before_rng[0])
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(policy.noise_key)), before_rng[1])
    assert sha(args.checkpoint) == proof['sha256']
    verification = dict(passed=True, identical_initial_full_state=True, paired_prefix_exact=True,
        checkpoint_unchanged=True, model_optimizer_unchanged=True, evaluation_rng_restored=True,
        max_reward_error=max_reward_error, primary_evaluation_unchanged=True)
    write(args.output / 'verification.json', verification)
    result = dict(completed=True, inference_only=True, training_source=cfg['source_commit'],
        evaluation_source=report_sha, checkpoint_step=step, conditions=summaries,
        seconds=time.monotonic() - started, verification=verification)
    write(args.output / 'result.json', result)
    try:
        import wandb
        parent = json.loads((args.run / 'wandb.json').read_text())
        identifier = hashlib.sha256(('paired-horizon:' + str(args.checkpoint) + ':' + str(args.extended_limit)).encode()).hexdigest()[:12]
        wb = wandb.init(entity='OptiQ', project='antmaze', id=identifier, resume='allow',
            group='antmaze-paired-horizon-diagnostic-20260925', job_type='evaluation',
            name=f"{cfg['task']}-T{cfg['temperature']:g}-paired-horizon700-{args.extended_limit}-{step}",
            config=dict(training_source=cfg['source_commit'], evaluation_source=report_sha,
                parent_run_id=parent['id'], parent_run_url=parent['url'],
                native_limit=700, supplementary_limit=args.extended_limit,
                inference_only=True, primary_evaluation_unchanged=True),
            settings=wandb.Settings(init_timeout=30))
        metrics = {'checkpoint_step': step, 'episodes_per_limit': args.episodes}
        for limit, summary in summaries.items():
            metrics[f'limit_{limit}/success_rate'] = summary['successes'] / args.episodes
            for side, n in summary['successful_routes'].items():
                metrics[f'limit_{limit}/successful_{side}'] = n
        wb.log(metrics)
        write(args.output / 'wandb.json', dict(id=identifier, url=wb.url, parent_run_id=parent['id']))
        wb.finish()
    except Exception as error:
        write(args.output / 'wandb-failure.json', dict(type=type(error).__name__, error=str(error), rollouts_verified=True))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
