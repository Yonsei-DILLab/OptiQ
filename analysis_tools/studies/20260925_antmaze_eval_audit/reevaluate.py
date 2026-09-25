"""Read-only AntMaze checkpoint evaluation under three start distributions.

The frozen training source supplies the original policy, maze and evaluator.
This script only restores a verified policy checkpoint and changes the reset
distribution used for evaluation; no training state is written back.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-profile', choices=('fixed-full', 'native-pose', 'random-xy'), required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--mode', choices=('policy', 'native'), default='policy')
    args = parser.parse_args()
    assert args.episodes > 0
    assert os.environ.get('JAX_PLATFORMS') == 'cpu' and not os.environ.get('CUDA_VISIBLE_DEVICES')
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / 'config.json').read_text())
    assert config['method'] == 'optiq' and config['task'] in ('v3', 'v4')
    assert config['dynamics_profile'] == 'control'
    assert config['eval_starts'] == 'upstream'
    assert config['reward_profile'] == 'progress100_euclidean_no_step_no_bonus'
    assert not config['dacer_enabled'] and not config['noveld_enabled']
    source_sha = subprocess.check_output(
        ['git', '-C', str(args.training_source), 'rev-parse', 'HEAD'], text=True).strip()
    assert source_sha == config['source_commit']
    checkpoint = args.run / 'checkpoint-final.pt'
    proof = json.loads((args.run / 'checkpoint-verification.json').read_text())
    assert proof['readback_verified'] and digest(checkpoint) == proof['sha256']
    sys.path.insert(0, str(args.training_source))

    import flax.serialization as serialization
    import gymnasium as gym
    import jax
    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from antmaze_experiments.critic_diagnostics import route_label
    from antmaze_experiments.learners import JaxLearner
    from antmaze_experiments.run import evaluate

    torch.set_num_threads(1)
    assert all(device.platform == 'cpu' for device in jax.devices())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['config'] == config and int(payload['step']) == int(proof['steps'])

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(29,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(8,), dtype=np.float32)

        def reset(self, **kwargs):
            raise RuntimeError('Inference descriptor only')

        def step(self, action):
            raise RuntimeError('Inference descriptor only')

    training = args.training_source / 'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py'
    spec = importlib.util.spec_from_file_location('frozen_trg_train', training)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native_config = OmegaConf.create(config['native'])
    native_config.output_root = str(args.output / 'constructor')
    model = module.runner.OptiQDIME('MlpPolicy', env=Descriptor(), cfg=native_config,
                                     model_save_path=None, save_every_n_steps=config['steps'])
    policy = model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state)
    restored = serialization.from_bytes(template, payload['learner']['policy'])
    saved_tree = serialization.msgpack_restore(payload['learner']['policy'])

    def assert_same_tree(actual, saved):
        if isinstance(saved, dict):
            assert set(actual) == set(saved)
            for key in saved:
                assert_same_tree(actual[key], saved[key])
        elif isinstance(saved, (tuple, list)):
            assert len(actual) == len(saved)
            for left, right in zip(actual, saved):
                assert_same_tree(left, right)
        else:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(saved))

    assert_same_tree(serialization.to_state_dict(restored), saved_tree)
    policy.actor_state = restored['actor']
    policy.qf_state = restored['critic']
    policy.target_actor_state = restored['target_actor']
    parameter_digest = hashlib.sha256(serialization.to_bytes(restored)).hexdigest()
    policy.key = jax.random.PRNGKey(700000 + int(payload['step']))
    policy.noise_key = jax.random.PRNGKey(700000 + int(payload['step']))

    learner = JaxLearner.__new__(JaxLearner)
    learner.method = 'optiq'
    learner.model = model
    learner.reward_profile = config['reward_profile']
    learner.eval_fixed_starts = args.start_profile == 'fixed-full'
    learner.eval_random_starts = args.start_profile == 'random-xy'
    # The original evaluator records Q diagnostics when present. Their extra
    # policy draws do not affect actions, but are unnecessary for this audit.
    learner.dynamics_profile = None
    result = evaluate(learner, config['task'], args.output, int(payload['step']),
                      args.episodes, args.mode, fixed=learner.eval_fixed_starts)
    rollout_path = args.output / 'evaluations' / f"{int(payload['step']):010d}" / (
        args.mode + ('-fixed' if learner.eval_fixed_starts else '-natural')) / 'rollouts.npz'
    with np.load(rollout_path) as rollout:
        starts = rollout['initial_full_state']
        lengths = rollout['lengths']
        labels = [route_label(config['task'], xy[:int(length) + 1])[0]
                  for xy, length in zip(rollout['xy'], lengths)]
        from collections import Counter
        routes = dict(Counter(labels))
        assert len(labels) == args.episodes
        assert np.isfinite(starts).all()
        unique_xy = len(np.unique(starts[:, :2], axis=0))
        unique_full = len(np.unique(starts, axis=0))
        if args.start_profile == 'fixed-full':
            assert unique_xy == unique_full == 1
        elif args.start_profile == 'native-pose':
            np.testing.assert_array_equal(starts[:, :2], np.zeros((args.episodes, 2)))
            assert unique_xy == 1 and unique_full > 1
        else:
            assert unique_xy > 1 and unique_full > 1
    after = hashlib.sha256(serialization.to_bytes(dict(
        actor=policy.actor_state, critic=policy.qf_state,
        target_actor=policy.target_actor_state))).hexdigest()
    assert parameter_digest == after and digest(checkpoint) == proof['sha256']
    report = dict(training_source=source_sha, checkpoint_sha256=proof['sha256'],
                  checkpoint_step=int(payload['step']), mode=args.mode,
                  start_profile=args.start_profile, episodes=args.episodes,
                  routes=routes, unique_xy_starts=unique_xy,
                  unique_full_starts=unique_full, summary=result,
                  rollout_sha256=digest(rollout_path),
                  restored_parameters_unchanged=True)
    write(args.output / 'evaluation-audit.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
