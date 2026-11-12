"""Re-execute the 25 official DAPG action trajectories in the pinned sparse v2 MDP."""
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request
import numpy as np
from .run import TASKS, atomic, make_env

DAPG_REVISION = '9a1de23df6574e1b94e1da742e3326eb1f61c013'
ROOT = Path('/workspace/heechan-adroit/datasets')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    expected = json.loads((Path(__file__).parent/'demo-source.json').read_text())
    for task in TASKS:
        path = ROOT/(task+'-v0_demos.pickle')
        url = f'https://raw.githubusercontent.com/aravindr93/hand_dapg/{DAPG_REVISION}/dapg/demonstrations/{path.name}'
        packed = ROOT/(task+'-dapg-actions.npz')
        if packed.exists():
            assert hashlib.sha256(packed.read_bytes()).hexdigest() == expected['action_pack_sha256'][packed.name]
            # The small transfer pack preserves every executed float32 action and
            # initial state. Raw official pickle hashes were checked locally.
            with np.load(packed, allow_pickle=False) as data:
                keys = json.loads(str(data['state_keys_json']))
                demos = [dict(actions=data[f'actions_{i}'].copy(), observations=data[f'observations_{i}'].copy(),
                    init_state_dict={k: data[f'state_{i}_{k}'].copy() for k in ks}) for i, ks in enumerate(keys)]
            digest = expected['sha256'][path.name]
        else:
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected['sha256'][path.name], 'Upstream demonstration bytes changed'
            # Only deserialize after checking the pinned official-repository hash.
            demos = pickle.loads(path.read_bytes())
        assert len(demos) == 25
        env = make_env(task)
        records = {k: [] for k in ('observations', 'actions', 'next_observations', 'dones', 'rewards')}
        episodes = []
        try:
            for index, demo in enumerate(demos):
                obs, _ = env.reset(seed=50000+index, options={'initial_state_dict': demo['init_state_dict']})
                successes = 0
                episode_start = len(records['actions'])
                initial_error = float(np.max(np.abs(obs-np.asarray(demo['observations'][0]))))
                for action in demo['actions']:
                    action = np.clip(action, -1., 1.).astype(np.float32)
                    # Replay the entire original demonstration, without adding artificial
                    # resets at the online TimeLimit. True termination masks are retained.
                    nxt, reward, terminated, _, info = env.unwrapped.step(action)
                    assert np.isfinite(obs).all() and np.isfinite(nxt).all() and np.isfinite(reward)
                    for k, value in zip(records, (obs, action, nxt, float(terminated), reward)):
                        records[k].append(value)
                    successes += int(info['success'])
                    obs = nxt
                    if terminated:
                        break
                episodes.append(dict(index=index, transitions=len(records['actions'])-episode_start,
                    success_steps=successes, initial_observation_max_error=initial_error))
            arrays = {k: np.asarray(v, np.float32) for k, v in records.items()}
            destination = ROOT/(task+'-human-sparse-v2.npz')
            np.savez_compressed(destination, **arrays)
            result = dict(task=task, env_name=TASKS[task], trajectories=25, transitions=len(arrays['actions']),
                source_url=url, source_sha256=digest, converted_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                source_revision=DAPG_REVISION, episodes=episodes, reward_values=np.unique(arrays['rewards']).tolist(),
                successful_trajectories=sum(e['success_steps'] > 0 for e in episodes),
                conversion='Full action replay from each supplied initial state in sparse v2; rewards and next observations recomputed; no filtering')
            atomic(destination.with_suffix('.json'), result)
            if result['successful_trajectories'] == 0:
                raise RuntimeError('No successful transferred demonstrations for '+task+'; inspect before training')
            results.append(result)
            print(json.dumps({k: result[k] for k in ('task', 'trajectories', 'transitions', 'successful_trajectories')}), flush=True)
        finally:
            env.close()
    atomic(ROOT/'verified-demonstrations.json', dict(passed=True, datasets=results))


if __name__ == '__main__':
    main()
