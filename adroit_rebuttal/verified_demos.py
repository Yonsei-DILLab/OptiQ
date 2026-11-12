"""Fail-closed native Sparse-v2 demonstration and on-disk replay validation."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import numpy as np
from .run import make_env, TASKS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(env):
    import gymnasium_robotics
    base = Path(gymnasium_robotics.__file__).parent
    # Pin both environment/reward implementation and every Adroit physics asset.
    files = {}
    for folder in (base/'envs/adroit_hand', base/'envs/assets/adroit_hand'):
        if folder.exists():
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                    files[str(p.relative_to(base))] = sha(p)
    assert any(p.endswith('.xml') for p in files), 'No physics XML assets fingerprinted'
    return dict(env_id=env.spec.id, max_episode_steps=env.spec.max_episode_steps,
        reward_type='sparse' if env.unwrapped.sparse_reward else 'dense', frame_skip=env.unwrapped.frame_skip,
        timestep=float(env.unwrapped.model.opt.timestep), gravity=env.unwrapped.model.opt.gravity.tolist(),
        observation_shape=list(env.observation_space.shape), action_shape=list(env.action_space.shape),
        mujoco=importlib.metadata.version('mujoco'), files=files)


def trajectory_hash(initial, actions):
    h = hashlib.sha256()
    for key, value in sorted(initial.items()):
        a = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(np.ascontiguousarray(actions).tobytes())
    return h.hexdigest()


def verify(manifest_path, replay=True):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    assert m['task'] in TASKS
    assert m['environment']['env_id'] == TASKS[m['task']]
    assert m['environment']['max_episode_steps'] == 200
    assert m['trajectories'] == m['successful_trajectories'] == 25
    assert len(m['episodes']) == 25 and len({e['trajectory_sha256'] for e in m['episodes']}) == 25
    folder = manifest_path.parent
    assert sha(folder/'collection-protocol.json') == m['collection_protocol_sha256']
    assert sha(folder/'original-state-audit.json') == m['original_state_audit_sha256']
    if m['provenance']['kind'] == 'expert-policy-demonstration':
        history_path = folder/'collection-history.jsonl'
        assert sha(history_path) == m['collection_history_sha256']
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        assert len(history) == m['attempted']
        assert [r['seed'] for r in history] == list(range(100000, 100000+len(history)))
        assert [r['seed'] for r in history if r['successful'] and r['complete_episode']][:25] == m['accepted_seeds']
        assert len({r['trajectory_sha256'] for r in history}) == len(history)
        assert all(sha(folder/r['file']) == r['sha256'] for r in history)
    for key in ('dataset', 'initial_states'):
        assert sha(folder/m[key]['file']) == m[key]['sha256'], key+' hash mismatch'
    # NPZ access decompresses a whole array on every __getitem__; load once
    # before stepping through 5,000 transitions.
    with np.load(folder/m['dataset']['file'], allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    with np.load(folder/m['initial_states']['file'], allow_pickle=False) as archive:
        states = {key:archive[key] for key in archive.files}
    assert all(np.isfinite(data[k]).all() for k in ('observations','actions','next_observations','rewards'))
    assert np.array_equal(data['dones'], data['terminated'])
    assert np.array_equal(data['timeouts'], data['truncated'] & ~data['terminated'])
    assert len(data['offsets']) == 26 and data['offsets'][0] == 0 and data['offsets'][-1] == len(data['actions'])
    env = make_env(m['task'])
    checks = []
    try:
        assert fingerprint(env) == m['environment'], 'Training environment/physics/reward differs from validation'
        for i, ep in enumerate(m['episodes']):
            start, end = map(int, data['offsets'][i:i+2])
            assert end-start == ep['length'] and 0 < ep['length'] <= 200
            assert np.all(data['trajectory_ids'][start:end] == i)
            initial = {k: states[f'state_{i}_{k}'].copy() for k in ep['state_keys']}
            assert trajectory_hash(initial, data['actions'][start:end]) == ep['trajectory_sha256']
            success = data['success'][start:end]
            assert success.any() and int(success.sum()) == ep['success_steps']
            assert bool(success[-1]) == ep['final_step_success']
            assert data['terminated'][end-1] or data['truncated'][end-1], 'Partial episode'
            assert not (data['terminated'][start:end-1] | data['truncated'][start:end-1]).any()
            max_error = 0.
            if replay:
                obs, _ = env.reset(seed=ep['seed'], options={'initial_state_dict': initial})
                observed = []
                for j in range(start, end):
                    error = float(np.max(np.abs(obs-data['observations'][j])))
                    max_error = max(max_error, error)
                    assert np.allclose(obs, data['observations'][j], rtol=0, atol=1e-8), (i,j,'observation',error)
                    obs, reward, terminated, truncated, info = env.step(data['actions'][j])
                    error = float(np.max(np.abs(obs-data['next_observations'][j])))
                    max_error = max(max_error, error)
                    assert np.allclose(obs, data['next_observations'][j], rtol=0, atol=1e-8), (i,j,'next_observation',error)
                    assert reward == data['rewards'][j]
                    assert terminated == bool(data['terminated'][j]) and truncated == bool(data['truncated'][j])
                    assert bool(info['success']) == bool(data['success'][j])
                    observed.append(bool(info['success']))
                assert any(observed) and sum(observed) == ep['success_steps']
                assert observed[-1] == ep['final_step_success']
            checks.append(dict(index=i, passed=True, success_steps=int(success.sum()),
                final_step_success=bool(success[-1]), max_observation_error=max_error))
    finally:
        env.close()
    return dict(passed=True, task=m['task'], replayed=replay, trajectories=25,
                successful_trajectories=25, dataset_sha256=m['dataset']['sha256'],
                manifest_sha256=sha(manifest_path), provenance=m['provenance'], episodes=checks)


def resolve(task, spec):
    if task not in ('hammer', 'pen') and 'demonstration_manifest' not in spec:
        return Path(f'/workspace/heechan-adroit/datasets/{task}-human-sparse-v2.npz'), None
    assert 'demonstration_manifest' in spec and 'demonstration_manifest_sha256' in spec, 'Verified 25/25 dataset is mandatory'
    path = Path(spec['demonstration_manifest'])
    assert sha(path) == spec['demonstration_manifest_sha256']
    m = json.loads(path.read_text())
    assert m['passed'] and m['task'] == task and m['saved_dataset_replay']['passed']
    assert sha(path.parent/'saved-replay-verification.json') == m['saved_dataset_replay']['report_sha256']
    proof = verify(path, replay=True)
    assert proof['passed']
    return path.parent/m['dataset']['file'], proof
