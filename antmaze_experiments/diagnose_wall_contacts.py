"""Read-only paired policy rollout with physical wall-contact instrumentation."""
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


def penetrating_wall_contacts(physics, wall_ids):
    """Exclude floor/Ant self contacts and positive-distance near contacts."""
    count, minimum = 0, 0.
    data, model = physics.sim.data, physics.model
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        wall1, wall2 = g1 in wall_ids, g2 in wall_ids
        if wall1 == wall2 or float(c.dist) > 0:
            continue
        other = g2 if wall1 else g1
        if int(model.geom_bodyid[other]) <= 0:
            continue
        count += 1
        minimum = min(minimum, float(c.dist))
    return count, minimum


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--cpu-offset', type=int, default=48)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 100
    assert not os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('JAX_PLATFORMS') == 'cpu'
    report_source = Path(__file__).resolve().parents[1]
    report_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=report_source, text=True).strip()
    cfg = json.loads((args.run / 'config.json').read_text())
    assert cfg['method'] == 'optiq' and cfg['task'] == 'v4'
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
        states = dict(actor=policy.actor_state, critic=policy.qf_state,
                      target_actor=policy.target_actor_state, entropy=model.ent_coef_state)
        if model.regulator_enabled:
            states.update(regulator_log_alpha=model.regulator_log_alpha,
                          regulator_optimizer=model.regulator_state)
        return hashlib.sha256(fs.to_bytes(states)).hexdigest()

    before = state_digest()
    before_rng = np.asarray(jax.random.key_data(policy.key)).copy(), np.asarray(jax.random.key_data(policy.noise_key)).copy()
    learner = JaxLearner.__new__(JaxLearner)
    learner.method, learner.model, learner.reward_profile = 'optiq', model, cfg['reward_profile']
    provenance = dict(training_source=cfg['source_commit'], evaluation_source=report_sha,
        parent_run=str(args.run), checkpoint=str(args.checkpoint), checkpoint_sha256=proof['sha256'],
        checkpoint_step=step, checkpoint_updates=int(payload['updates']), training_seed=cfg['seed'], task=cfg['task'],
        episodes_per_condition=args.episodes, native_limit=700,
        instrumentation='pre-action qpos/qvel and penetrating Ant-versus-block contacts',
        reset='original identical full state at origin', mode='policy', conditional_sigma=True,
        random_latent=True, external_dacer_noise=False, intrinsic_reward=False, inference_only=True,
        primary_evaluation_unchanged=True, goal_radius_unchanged=True,
        paired_rng='700000+checkpoint_step; restored around plain and instrumented conditions',
        limitation='Contact and low progress correlations do not establish a causal remedy. Diagnostic40episode CPU evaluation does not replace final100episode primary results.',
        training_config=cfg)
    write(args.output / 'provenance.json', provenance)

    started = time.monotonic()
    all_results = {}
    count = min(20, args.episodes)
    geometry = None
    for condition in ('plain', 'instrumented'):
        records = []
        env = vector(cfg['task'], count, seed=87231, asynchronous=False, fixed=True,
                     reward_profile=cfg['reward_profile'], random_init=False)
        try:
            wall_sets = []
            for one in env.envs:
                assert one.env._max_episode_steps == 700
                model = one.physics_env.model
                names = [model.geom_id2name(i) for i in range(int(model.ngeom))]
                walls = {i for i, name in enumerate(names) if name and name.startswith('block_')}
                assert walls and all(model.geom_bodyid[i] == 0 for i in walls)
                wall_sets.append(walls)
                description = dict(names=names, wall_ids=sorted(walls),
                    geom_pos=np.asarray(model.geom_pos).tolist(), geom_size=np.asarray(model.geom_size).tolist(),
                    timestep=float(model.opt.timestep), frame_skip=int(one.physics_env.frame_skip))
                if geometry is None:
                    geometry = description
                else:
                    assert geometry == description
            with evaluation_rng(learner, 700000 + step):
                for batch in range((args.episodes + count - 1) // count):
                    obs = env.reset()
                    common = env.envs[0].initial
                    for i, one in enumerate(env.envs):
                        one.initial = common
                        obs[i] = one.restore(common)
                    initial = [np.r_[one.state()['qpos'], one.state()['qvel']] for one in env.envs]
                    active = np.arange(count) + batch * count < args.episodes
                    tracks = [[o[:2].copy()] for o in obs]
                    action_tracks = [[] for _ in obs]
                    poses, velocities = [[] for _ in obs], [[] for _ in obs]
                    contact_counts, contact_distances = [[] for _ in obs], [[] for _ in obs]
                    rewards, goals, lengths = np.zeros(count), np.zeros(count, int), np.zeros(count, int)
                    for _ in range(700):
                        if condition == 'instrumented':
                            for i in np.flatnonzero(active):
                                physics = env.envs[i].physics_env
                                data = physics.sim.data
                                np.testing.assert_array_equal(np.asarray(data.qpos[:2], dtype=np.float32), obs[i, :2])
                                poses[i].append(np.array(data.qpos, copy=True))
                                velocities[i].append(np.array(data.qvel, copy=True))
                                n, depth = penetrating_wall_contacts(physics, wall_sets[i])
                                contact_counts[i].append(n)
                                contact_distances[i].append(depth)
                        actions = learner.act(obs, 'policy')
                        nxt, r, done, infos = env.step(actions)
                        final, _ = transition(nxt, done, infos)
                        for i in np.flatnonzero(active):
                            tracks[i].append(final[i, :2].copy())
                            action_tracks[i].append(actions[i].copy())
                            rewards[i] += r[i]
                            lengths[i] += 1
                            if done[i]:
                                goals[i] = int(infos[i].get('success', 0))
                                active[i] = False
                        obs = nxt
                        if not active.any():
                            break
                    for i in range(min(count, args.episodes - batch * count)):
                        item = dict(xy=np.asarray(tracks[i]), actions=np.asarray(action_tracks[i]),
                            returns=float(rewards[i]), goal=int(goals[i]), length=int(lengths[i]), initial=initial[i])
                        if condition == 'instrumented':
                            item.update(qpos=np.asarray(poses[i]), qvel=np.asarray(velocities[i]),
                                wall_count=np.asarray(contact_counts[i]), wall_min_distance=np.asarray(contact_distances[i]))
                            assert len(item['qpos']) == item['length']
                        records.append(item)
                    write(args.output/'progress.json', dict(condition=condition, completed_episodes=len(records),
                        episodes_per_condition=args.episodes, seconds=time.monotonic()-started))
        finally:
            env.close()
        all_results[condition] = records
    for plain, instrumented in zip(all_results['plain'], all_results['instrumented']):
        for name in ('xy', 'actions', 'initial'):
            np.testing.assert_array_equal(plain[name], instrumented[name])
        for name in ('returns', 'goal', 'length'):
            assert plain[name] == instrumented[name], name
    summaries = {}
    episode_rows = []
    max_error = 0.
    for condition, records in all_results.items():
        def padded(name, extra=0):
            shape = (args.episodes, 700 + extra) + records[0][name].shape[1:]
            value = np.full(shape, np.nan, dtype=np.float64)
            for i, record in enumerate(records):
                value[i, :len(record[name])] = record[name]
            return value
        arrays = dict(xy=padded('xy', 1), actions=padded('actions'),
            returns=np.array([r['returns'] for r in records]), goals=np.array([r['goal'] for r in records]),
            lengths=np.array([r['length'] for r in records]), initial_full_state=np.asarray([r['initial'] for r in records]))
        np.testing.assert_array_equal(arrays['initial_full_state'], np.repeat(arrays['initial_full_state'][:1], args.episodes, axis=0))
        np.testing.assert_array_equal(arrays['initial_full_state'][:, :2], np.zeros((args.episodes, 2)))
        routes, successes = {}, {}
        if condition == 'instrumented':
            arrays.update({k: padded(k) for k in ('qpos', 'qvel', 'wall_count', 'wall_min_distance')})
        for i, r in enumerate(records):
            path, n = r['xy'], r['length']
            route, _ = route_label(cfg['task'], path)
            routes[route] = routes.get(route, 0) + 1
            if r['goal']:
                successes[route] = successes.get(route, 0) + 1
            profile = cfg['reward_profile']
            assert profile in PROFILES and not bonus_enabled(profile)
            expected = progress_scale(profile) * (distance(path[0], cfg['task'], profile)-distance(path[-1], cfg['task'], profile)) - step_cost(profile)*n
            max_error = max(max_error, abs(r['returns'] - float(expected)))
            if condition == 'instrumented':
                tail = min(200, n)
                episode_rows.append(dict(episode=i, route=route, goal=r['goal'], steps=n,
                    wall_contact_fraction=float(np.mean(r['wall_count'] > 0)),
                    last200_wall_contact_fraction=float(np.mean(r['wall_count'][-tail:] > 0)),
                    last200_net_displacement=float(np.linalg.norm(path[-1]-path[-tail-1])),
                    last200_path_length=float(np.linalg.norm(np.diff(path[-tail-1:], axis=0),axis=-1).sum()),
                    last200_planar_speed_rms=float(np.sqrt(np.mean(np.sum(r['qvel'][-tail:,:2]**2,axis=1)))),
                    last200_mean_height=float(r['qpos'][-tail:,2].mean()),
                    final_geodesic=float(distance(path[-1], cfg['task'], profile)),
                    final_xy=path[-1].tolist()))
        raw = args.output/(condition+'.npz')
        np.savez_compressed(raw, **arrays)
        summaries[condition] = dict(episodes=args.episodes, routes=routes, successful_routes=successes,
            successes=sum(bool(r['goal']) for r in records), raw=raw.name, sha256=sha(raw))
    assert max_error < .01 and state_digest() == before
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(policy.key)), before_rng[0])
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(policy.noise_key)), before_rng[1])
    assert sha(args.checkpoint) == proof['sha256']
    verification = dict(passed=True, checkpoint_unchanged=True, model_optimizer_unchanged=True,
        restored_policy_exact=True,
        evaluation_rng_restored=True, paired_xy_actions_returns_exact=True,
        identical_initial_full_state=True, native_horizon_unchanged=True,
        contact_definition='dist<=0; exactly one block_* geom, other belongs to non-world body; floor excluded; pre-action readout',
        max_reward_error=max_error)
    result = dict(completed=True, inference_only=True, training_source=cfg['source_commit'],
        evaluation_source=report_sha, checkpoint_step=step, conditions=summaries,
        geometry=geometry, episode_diagnostics=episode_rows, verification=verification,
        seconds=time.monotonic()-started)
    write(args.output/'verification.json', verification)
    write(args.output/'result.json', result)
    try:
        import wandb
        parent = json.loads((args.run/'wandb.json').read_text())
        identifier = hashlib.sha256(('wall-contact:' + str(args.checkpoint)).encode()).hexdigest()[:12]
        wb = wandb.init(entity='OptiQ', project='antmaze', id=identifier, resume='allow',
            group='antmaze-wall-contact-diagnostic-20260925', job_type='evaluation',
            name=f"v4-teacherfloor{cfg['teacher_std_floor_override']:g}-wall-contact-{step}",
            config=dict(training_source=cfg['source_commit'], evaluation_source=report_sha,
                parent_run_id=parent['id'], parent_run_url=parent['url'],
                episodes=args.episodes, native_horizon=700, paired_instrumentation=True,
                inference_only=True, primary_evaluation_unchanged=True),
            settings=wandb.Settings(init_timeout=30))
        metrics = {'checkpoint_step':step, 'success_rate':summaries['instrumented']['successes']/args.episodes}
        for route in sorted({r['route'] for r in episode_rows}):
            rows = [r for r in episode_rows if r['route']==route]
            metrics[f'route/{route}/episodes'] = len(rows)
            for key in ('last200_wall_contact_fraction','last200_net_displacement',
                        'last200_planar_speed_rms','last200_mean_height','final_geodesic'):
                metrics[f'route/{route}/{key}'] = float(np.mean([r[key] for r in rows]))
        wb.log(metrics)
        write(args.output/'wandb.json',dict(id=identifier,url=wb.url,parent_run_id=parent['id']))
        wb.finish()
    except Exception as error:
        write(args.output/'wandb-failure.json',dict(type=type(error).__name__,error=str(error),rollouts_verified=True))
    print(json.dumps(dict(completed=True, conditions=summaries, verification=verification)),flush=True)


if __name__ == '__main__':
    main()
