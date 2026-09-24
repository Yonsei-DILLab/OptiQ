"""Fresh random-start inference from immutable historical OptiQ checkpoints.

Run as a script in a fresh process, with CPU-only JAX. Training-source imports
deliberately precede the reporting source, preserving each policy/environment.
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


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--family', choices=['legacy', 'official'], required=True)
    ap.add_argument('--task', choices=['v1', 'v2', 'v3', 'v4'], required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--training-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--evaluation-source', required=True)
    ap.add_argument('--episodes', type=int, required=True)
    ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--cpu-offset', type=int, default=0)
    ap.add_argument('--v1-origin-supplement', action='store_true',
                    help='Supplementary identical-origin probe; never replaces v1 native random evaluation')
    ap.add_argument('--evaluation-seed', type=int, default=20260924)
    ap.add_argument('--checkpoint', type=Path,
                    help='Optional immutable official policy checkpoint inside --run')
    a = ap.parse_args()
    assert a.episodes > 0 and a.episodes % a.batch == 0
    if a.v1_origin_supplement:
        assert a.family == 'official' and a.task == 'v1'
    if a.checkpoint is not None:
        assert a.family == 'official' and a.checkpoint.is_relative_to(a.run)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[a.cpu_offset:a.cpu_offset + 4])
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.training_source))
    import numpy as np
    import torch
    import jax
    import flax.serialization as fs
    from omegaconf import OmegaConf
    import gymnasium as gym
    torch.set_num_threads(1)
    assert all(d.platform == 'cpu' for d in jax.devices())
    cfg = json.loads((a.run / 'config.json').read_text())
    training_sha = subprocess.check_output(['git', '-C', str(a.training_source), 'rev-parse', 'HEAD'], text=True).strip()
    assert training_sha == cfg['source_commit'] and cfg['method'] == 'optiq' and cfg['task'] == a.task
    if a.family == 'legacy':
        checkpoint = a.run / 'resume/step_0001000000/state.pt'
        proof = json.loads(checkpoint.with_name('manifest.json').read_text())
        expected_sha = proof['files']['state.pt']['sha256']
        from antmaze.multimodal.env import make_env, GOALS
        import mujoco
        envs = [make_env(a.task) for _ in range(a.batch)]
    else:
        checkpoint = a.checkpoint or a.run / 'checkpoint-final.pt'
        proof_path = (checkpoint.parent/'verification.json' if checkpoint.name == 'policy.pt'
                      else a.run/'checkpoint-verification.json')
        proof = json.loads(proof_path.read_text())
        assert proof['readback_verified']
        expected_sha = proof['sha256']
        from antmaze_experiments.envs import make_one
        envs = [make_one(a.task, 420000 + i, reward_profile='dense', random_init=True)
                for i in range(a.batch)]
        GOALS = {a.task: np.asarray(envs[0].physics_env.target_goal).reshape(-1, 2)}
    assert sha(checkpoint) == expected_sha
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    checkpoint_step = int(payload.get('step', cfg['steps']))
    if a.family == 'official':
        assert payload['config'] == cfg
        assert checkpoint_step == int(proof.get('step', proof.get('steps')))
        assert cfg['reward_profile'] == 'dense'

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
        def reset(self, **kwargs):
            raise RuntimeError('Inference descriptor cannot collect data')
        def step(self, action):
            raise RuntimeError('Inference descriptor cannot collect data')

    spec = importlib.util.spec_from_file_location('frozen_trg_train',
        a.training_source / 'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native_cfg = OmegaConf.create(cfg['native'])
    # Only constructor output destination is redirected. The actual trained
    # architecture, policy bounds, transforms and sampler remain frozen.
    native_cfg.output_root = str(a.output / 'constructor')
    model = module.runner.OptiQDIME('MlpPolicy', env=Descriptor(), cfg=native_cfg,
                                  model_save_path=None, save_every_n_steps=cfg['steps'])
    policy = model.policy
    if a.family == 'legacy':
        from antmaze.multimodal.resume import unpack_flax
        saved = payload['model']['policy']
        for key in ('actor_state', 'target_actor_state', 'qf_state'):
            setattr(policy, key, unpack_flax(getattr(policy, key), saved[key]))
        restored = fs.to_state_dict(policy.actor_state)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x,y in
                   zip(jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(saved['actor_state'])))
        updates = payload['model']['updates']
    else:
        template = dict(actor=policy.actor_state, critic=policy.qf_state, target_actor=policy.target_actor_state)
        saved = fs.from_bytes(template, payload['learner']['policy'])
        policy.actor_state, policy.qf_state, policy.target_actor_state = saved['actor'], saved['critic'], saved['target_actor']
        # Msgpack map ordering/array wrappers may change after restoration;
        # compare every named value, not the container's serialized bytes.
        original = fs.msgpack_restore(payload['learner']['policy'])
        def equal_tree(actual, expected):
            if isinstance(expected, dict):
                assert set(actual) == set(expected)
                for key in expected:
                    equal_tree(actual[key], expected[key])
            elif isinstance(expected, (list, tuple)):
                assert len(actual) == len(expected)
                for x, y in zip(actual, expected):
                    equal_tree(x, y)
            else:
                x, y = np.asarray(actual), np.asarray(expected)
                assert x.shape == y.shape and x.dtype == y.dtype
                np.testing.assert_array_equal(x, y)
        equal_tree(fs.to_state_dict(saved), original)
        updates = payload['updates']
    states = dict(actor=policy.actor_state, critic=policy.qf_state, target_actor=policy.target_actor_state)
    before = hashlib.sha256(fs.to_bytes(states)).hexdigest()
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(states))
    del payload
    horizon = 500 if a.task in ('v1', 'v2') else 700
    seed = [a.evaluation_seed, 90177, int(a.task[1:])]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    positions = (np.zeros((a.episodes, 2)) if a.v1_origin_supplement
                 else rng.uniform(-2., 2., size=(a.episodes, 2)))
    policy_seeds = rng.integers(1, 2**30, size=a.episodes // a.batch)
    env_seeds = rng.integers(1, 2**30, size=a.episodes)
    provenance = dict(family=a.family, task=a.task, training_source=training_sha,
        evaluation_source=a.evaluation_source, checkpoint=str(checkpoint), checkpoint_sha256=expected_sha,
        checkpoint_steps=checkpoint_step, learner_updates=updates, training_seed=cfg['seed'],
        episodes_per_mode=a.episodes, batch=a.batch,
        start_distribution=('identical original origin [0,0], pose and velocity; supplementary only'
                            if a.v1_origin_supplement else 'iid xy uniform[-2,2]; original initial pose/velocity'),
        supplementary_origin_probe=a.v1_origin_supplement,
        replaces_primary_evaluation=False,
        evaluation_override_v1=a.v1_origin_supplement,
        evaluation_override_v234=a.task != 'v1', seed_sequence=seed,
        modes=['policy', 'native'], policy='fresh random z + conditional sigma', native='fresh random z mu-only',
        external_exploration_noise=False, intrinsic_reward=False, training_config=cfg,
        runtime_devices=[str(d) for d in jax.devices()], restored_policy_exact=True,
        script_sha256=sha(Path(__file__)))
    write(a.output / 'provenance.json', provenance)
    started = time.monotonic()
    results, initial_states = {}, {}
    try:
        for mode in ('policy', 'native'):
            xy = np.full((a.episodes, horizon + 1, 2), np.nan, np.float32)
            actions = np.full((a.episodes, horizon, 8), np.nan, np.float32)
            full_states, lengths = [], np.zeros(a.episodes, np.int64)
            returns, goals = np.zeros(a.episodes), np.zeros(a.episodes, np.int64)
            for start in range(0, a.episodes, a.batch):
                obs = []
                for i, env in enumerate(envs):
                    ix = start + i
                    if a.family == 'legacy':
                        env.reset(seed=int(env_seeds[ix]), options={'fixed_start': True})
                        env.data.qpos[:2] = positions[ix]
                        mujoco.mj_forward(env.model, env.data)
                        o = env.observation()
                        st = np.r_[env.data.qpos.copy(), env.data.qvel.copy()]
                    else:
                        env.reset()
                        state = env.state()
                        state['qpos'][:2] = positions[ix]
                        o = env.restore(state)
                        st = np.r_[state['qpos'], state['qvel']]
                    obs.append(o)
                    full_states.append(st)
                    xy[ix, 0] = o[:2]
                obs = np.asarray(obs)
                active = np.ones(a.batch, bool)
                key = jax.random.PRNGKey(int(policy_seeds[start // a.batch]))
                while active.any():
                    key, action_key = jax.random.split(key)
                    act = np.asarray(policy.sample_action(policy.actor_state, obs, action_key,
                        deterministic=False, sample_conditional_noise=mode == 'policy'))
                    assert act.shape == (a.batch, 8) and np.isfinite(act).all()
                    assert np.max(np.abs(act)) <= 1.000001
                    for i in np.flatnonzero(active):
                        ix, t = start + i, lengths[start + i]
                        actions[ix, t] = act[i]
                        if a.family == 'legacy':
                            obs[i], reward, term, trunc, info = envs[i].step(act[i])
                            done, goal = term or trunc, info['goal_id']
                        else:
                            obs[i], reward, done, info = envs[i].step(act[i])
                            goal = int(info.get('success', 0))
                        xy[ix, t + 1] = obs[i, :2]
                        lengths[ix] += 1
                        returns[ix] += reward
                        goals[ix] = goal
                        active[i] = not done
                write(a.output / 'progress.json', dict(mode=mode, episodes_done=start+a.batch,
                      episodes_per_mode=a.episodes, seconds=time.monotonic()-started))
            full_states = np.asarray(full_states)
            np.testing.assert_allclose(full_states[:, :2], positions, atol=1e-12)
            if a.v1_origin_supplement:
                np.testing.assert_array_equal(full_states, np.repeat(full_states[:1], a.episodes, axis=0))
            else:
                assert len(np.unique(full_states[:, :2], axis=0)) == a.episodes
            initial_states[mode] = full_states
            distance = np.linalg.norm(xy[:, :, None, :] - np.asarray(GOALS[a.task])[None, None, :, :], axis=-1)
            np.testing.assert_array_equal(np.nanmin(distance, axis=(1, 2)) <= .50002, goals > 0)
            np.testing.assert_allclose(-np.nansum(distance[:, 1:].min(axis=-1), axis=1), returns, rtol=2e-6, atol=.004)
            np.savez_compressed(a.output / f'{mode}.npz', xy=xy, actions=actions, lengths=lengths,
                returns=returns, goal_ids=goals, initial_full_state=full_states,
                initial_positions=positions, policy_batch_seeds=policy_seeds, env_seeds=env_seeds)
            result = dict(episodes=a.episodes, successes=int((goals > 0).sum()),
                          goals={str(int(g)):int((goals == g).sum()) for g in np.unique(goals)},
                          mean_return=float(returns.mean()),
                          random_initial_states=0 if a.v1_origin_supplement else a.episodes,
                          unique_full_initial_states=len(np.unique(full_states, axis=0)),
                          supplementary_origin_probe=a.v1_origin_supplement)
            write(a.output / f'{mode}.json', result)
            results[mode] = result
            print(json.dumps(dict(family=a.family, task=a.task, mode=mode, **result)), flush=True)
        np.testing.assert_array_equal(initial_states['policy'], initial_states['native'])
        assert hashlib.sha256(fs.to_bytes(states)).hexdigest() == before
        assert sha(checkpoint) == expected_sha
        write(a.output / 'verification.json', dict(passed=True, checkpoint_unchanged=True,
            model_optimizer_unchanged=True, paired_random_initial_states=not a.v1_origin_supplement,
            paired_identical_origin_states=a.v1_origin_supplement,
            goals_and_dense_returns_recomputed=True, initial_xy_iid_uniform=not a.v1_origin_supplement,
            npz_sha256={p.name:sha(p) for p in a.output.glob('*.npz')}))
        write(a.output / 'result.json', dict(completed=True, seconds=time.monotonic()-started, results=results))
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
