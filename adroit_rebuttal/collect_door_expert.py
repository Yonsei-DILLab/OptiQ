"""Collect the first 25 successful native-v2 Door episodes from pinned DAPG expert.

The historical checkpoint's normalizers and deterministic evaluation mean are
preserved. Compatibility classes only deserialize obsolete MJRL/THNN names.
The expert is a separate data source, never a learner initialization.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch

CHECKPOINT_SHA256 = '288dd2131b905e40852d9eb032595bb9b3aa90b0502e9341e6b0e128b803478f'
CHECKPOINT_URL = 'https://raw.githubusercontent.com/aravindr93/hand_dapg/9a1de23df6574e1b94e1da742e3326eb1f61c013/dapg/policies/door-v0.pickle'


class HistoricalPolicy:
    pass


class HistoricalNetwork(torch.nn.Module):
    def forward(self, x):
        out = (x - self.in_shift) / (self.in_scale + 1e-8)
        for layer in self.fc_layers[:-1]:
            out = self.nonlinearity(layer(out))
        return self.fc_layers[-1](out) * self.out_scale + self.out_shift


class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('mjrl.policies.gaussian_mlp', 'MLP'):
            return HistoricalPolicy
        if (module, name) == ('mjrl.utils.fc_network', 'FCNetwork'):
            return HistoricalNetwork
        if (module, name) == ('torch.nn.backends.thnn', '_get_thnn_function_backend'):
            return lambda: None
        return super().find_class(module, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    args.output.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open('rb') as f:
        teacher = CompatibilityUnpickler(f).load().model.eval()
    assert teacher.layer_sizes == (39, 32, 32, 28)
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make('AdroitHandDoorSparse-v2')
    assert env.spec.max_episode_steps == 200
    rows = {k: [] for k in ('observations', 'actions', 'next_observations', 'rewards', 'dones', 'timeouts', 'trajectory_ids')}
    accepted, attempts = [], []
    initial_states = {}

    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        initial = {k: v.copy() for k,v in env.unwrapped.get_env_state().items()}
        transitions, successes = [], []
        with torch.no_grad():
            for t in range(200):
                raw = teacher(torch.as_tensor(obs, dtype=torch.float32)[None]).numpy()[0]
                assert np.isfinite(raw).all()
                action = np.clip(raw, -1, 1).astype(np.float32)
                nxt, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(nxt).all()
                transitions.append((obs.copy(), action, nxt.copy(), reward, float(terminated), float(truncated)))
                successes.append(bool(info['success']))
                obs = nxt
                if terminated or truncated:
                    break
        return initial, transitions, dict(seed=seed, length=len(transitions), successful=any(successes),
            success_steps=sum(successes), first_success=next((i+1 for i,v in enumerate(successes) if v),None),
            return_=sum(t[3] for t in transitions))

    try:
        for seed in range(100000, 100500):
            initial, trajectory, record = rollout(seed)
            attempts.append(record)
            if record['successful']:
                index = len(accepted)
                for k,v in initial.items():
                    initial_states[f'state_{index}_{k}'] = v
                for transition in trajectory:
                    for k,v in zip(rows, (*transition,index)):
                        rows[k].append(v)
                accepted.append(record)
                print(json.dumps(dict(accepted=len(accepted), attempted=len(attempts), **record)), flush=True)
            if len(accepted) == 25:
                break
        manifest = dict(source='Official DAPG pretrained RL expert, freshly rolled out in native sparse v2',
            checkpoint_url=CHECKPOINT_URL, checkpoint_sha256=CHECKPOINT_SHA256,
            teacher_architecture=[32,32], teacher_input_output_normalizers_preserved=True,
            teacher_evaluation='deterministic Gaussian mean, clipped by native action bounds',
            collection_rule='First 25 successes in ascending seeds 100000..100499; full native 200-step episodes; no ranking',
            same_native_reset_distribution_as_online=True, teacher_not_used_for_learner_initialization=True,
            expert_pretraining='External pretrained RL expert; original training reward must be reported separately from sparse learner reward.',
            accepted=accepted, attempts=attempts, trajectories=len(accepted))
        if len(accepted) != 25:
            manifest['passed'] = False
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            raise RuntimeError('The pinned expert did not produce 25 successful native-v2 demonstrations in the predefined attempt budget')
        # Verify actual saved actions replay to the same transitions and sparse rewards.
        offset = 0
        for i,record in enumerate(accepted):
            initial = {k: initial_states[f'state_{i}_{k}'] for k in ('qpos','qvel','door_body_pos')}
            obs,_ = env.reset(seed=record['seed'],options={'initial_state_dict':initial})
            for t in range(record['length']):
                j = offset+t
                assert np.allclose(obs,rows['observations'][j],rtol=0,atol=1e-8)
                obs,r,done,timeout,_ = env.step(rows['actions'][j])
                assert np.allclose(obs,rows['next_observations'][j],rtol=0,atol=1e-8)
                assert r == rows['rewards'][j] and done == bool(rows['dones'][j]) and timeout == bool(rows['timeouts'][j])
            offset += record['length']
        destination = args.output/'door-expert25-sparse-v2.npz'
        np.savez_compressed(destination, **{k:np.asarray(v,np.int32 if k=='trajectory_ids' else np.float32) for k,v in rows.items()})
        np.savez_compressed(args.output/'initial-states.npz', **initial_states)
        holdout = [rollout(seed)[2] for seed in range(200000,200100)]
        manifest.update(passed=True, transitions=offset, sparse_reward_values=np.unique(rows['rewards']).tolist(),
            dataset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            saved_action_replay_verified=True, heldout_teacher_episodes=holdout,
            heldout_teacher_success_rate=sum(r['successful'] for r in holdout)/100)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('attempts','accepted','heldout_teacher_episodes')}),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
