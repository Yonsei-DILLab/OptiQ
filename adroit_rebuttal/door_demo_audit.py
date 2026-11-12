"""Reconstruct and validate DAPG Door initial states without changing training.

All 25 trajectories are retained in source order. Reconstruction uses only the
first recorded observation, never trajectory reward or future success. The
unobserved first slide joint is recovered from palm position; supplied initial
velocities are retained. Full replay is compared with the unmodified importer.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def restore(env, state, observation, recover_translation):
    state = copy.deepcopy(state)
    qpos = state['qpos']
    assert qpos.shape == (30,) and observation.shape == (39,)
    qpos[1:-2] = observation[:27]
    qpos[-1] = observation[27]
    qpos[-2] = observation[28]
    env.reset(seed=50000, options={'initial_state_dict': state})
    if recover_translation:
        import mujoco
        assert env.unwrapped.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE
        palm = env.unwrapped._get_obs()[29:32].copy()
        probe = copy.deepcopy(state)
        probe['qpos'][0] += 1e-4
        env.unwrapped.set_env_state(probe)
        derivative = (env.unwrapped._get_obs()[29:32] - palm) / 1e-4
        assert np.dot(derivative, derivative) > .9
        state['qpos'][0] += np.dot(observation[29:32] - palm, derivative) / np.dot(derivative, derivative)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import gymnasium as gym
    import gymnasium_robotics
    import mujoco
    gym.register_envs(gymnasium_robotics)
    env = gym.make('AdroitHandDoorSparse-v2')
    data = np.load(args.pack, allow_pickle=False)
    keys = json.loads(str(data['state_keys_json']))
    assert len(keys) == 25
    joint_names = [mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(env.unwrapped.model.njnt)]
    report = dict(environment='AdroitHandDoorSparse-v2', mujoco=mujoco.__version__,
                  source_pack_sha256=hashlib.sha256(args.pack.read_bytes()).hexdigest(),
                  joint_names=joint_names, profiles={},
                  velocity_limitation='Initial velocities retained from supplied state; observations do not identify all velocities.',
                  selection='All 25 trajectories, no reward-based filtering or success-based reconstruction.')
    try:
        for profile in ('supplied_state', 'observed_qpos', 'observed_qpos_and_palm'):
            arrays = {k: [] for k in ('observations', 'actions', 'next_observations', 'rewards', 'dones', 'timeouts', 'trajectory_ids')}
            episodes = []
            for i, state_keys in enumerate(keys):
                state = {k: data[f'state_{i}_{k}'].copy() for k in state_keys}
                original_observations = data[f'observations_{i}']
                if profile != 'supplied_state':
                    state = restore(env, state, original_observations[0], profile == 'observed_qpos_and_palm')
                obs, _ = env.reset(seed=50000+i, options={'initial_state_dict': state})
                initial_error = float(np.max(np.abs(obs-original_observations[0])))
                state_path = args.output / (profile+'-initial-'+str(i)+'.npz')
                np.savez_compressed(state_path, **state)
                success, errors = [], []
                start = len(arrays['actions'])
                for t, raw_action in enumerate(data[f'actions_{i}']):
                    action = np.clip(raw_action, -1, 1).astype(np.float32)
                    errors.append(float(np.max(np.abs(obs-original_observations[t]))))
                    nxt, reward, terminated, truncated, info = env.unwrapped.step(action)
                    assert not truncated
                    assert np.isfinite(obs).all() and np.isfinite(nxt).all() and np.isfinite(reward)
                    for k, v in zip(arrays, (obs, action, nxt, reward, float(terminated), 0., i)):
                        arrays[k].append(v)
                    success.append(bool(info['success']))
                    obs = nxt
                    if terminated:
                        break
                first = next((t+1 for t, ok in enumerate(success) if ok), None)
                ep = dict(index=i, start=start, transitions=len(success), initial_observation_max_error=initial_error,
                          success_steps=sum(success), successful=any(success), success_within_200=any(success[:200]),
                          first_success=first, replay_observation_error_mean=float(np.mean(errors)),
                          replay_observation_error_max=max(errors), restored_qpos=state['qpos'].tolist())
                episodes.append(ep)
            destination = args.output/(profile+'.npz')
            np.savez_compressed(destination, **{k: np.asarray(v, np.int32 if k == 'trajectory_ids' else np.float32) for k,v in arrays.items()})
            summary = dict(trajectories=25, transitions=len(arrays['actions']), episodes=episodes,
                           successful_trajectories=sum(e['successful'] for e in episodes),
                           successful_within_200=sum(e['success_within_200'] for e in episodes),
                           max_initial_observation_error=max(e['initial_observation_max_error'] for e in episodes),
                           sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                           full_source_horizon_retained=True)
            report['profiles'][profile] = summary
            (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(profile=profile, **{k:v for k,v in summary.items() if k!='episodes'})), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
