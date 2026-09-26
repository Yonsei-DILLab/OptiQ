"""Train the T=5, seed-0 DrAC PointMaze profile without baseline dependencies."""
import argparse
from contextlib import contextmanager
import json
import random
from pathlib import Path

import flax.serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3.common.logger import configure

from ibolt.algorithm import IBOLT
from ibolt.runtime import provenance
from train import compose_config
from .batch import TaskBatch
from .drac_paper import PaperPointMaze, HORIZONS, GOAL_COUNTS
from .replay import Replay
from .evaluation import evaluate as evaluate_batched


class SpaceOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

    def reset(self, **kwargs):
        raise RuntimeError('TaskBatch owns the simulator')

    def step(self, action):
        raise RuntimeError('TaskBatch owns the simulator')


def configuration(seed=0, temperature=5., batch_size=4096, maze='simple'):
    cfg = compose_config(['benchmark=ant', f'seed={seed}', 'dacer.enabled=false',
                          'wandb.activate=false'])
    cfg.alg.actor.temperature = temperature
    cfg.alg.learning_starts = cfg.alg.actor.learning_starts = 0
    cfg.alg.batch_size = batch_size
    cfg.task = f'pointmaze-{maze}'
    cfg.env_name = f'DrAC-PointMaze-{maze}'
    cfg.diagnostic_interval = 0
    return cfg


def expected_updates(steps=1_000_192, warmup=8192, num_envs=256, updates=16):
    if steps <= warmup or steps % num_envs or warmup % num_envs:
        raise ValueError('Budgets must be full collector batches, beyond warmup')
    return (steps - warmup) // num_envs * updates


class Agent:
    method = 'optiq'  # Historical evaluator's mu-only validation key.
    def __init__(self, cfg, output):
        self.cfg = cfg
        self.model = IBOLT('MlpPolicy', SpaceOnlyEnv(), cfg=cfg,
                           model_save_path=None, save_every_n_steps=0)
        self.model.set_logger(configure(str(output / 'learner'), ['csv']))
        self.replay = Replay(seed=cfg.seed)
        self.model.replay_buffer = self.replay

    def act(self, observations, mean_only=False, mode=None):
        if mode is not None:
            mean_only = mode == 'mu_only'
        policy = self.model.policy
        policy.reset_noise()
        actions = policy.sample_action(policy.actor_state, jnp.asarray(observations),
            policy.noise_key, deterministic=False, sample_conditional_noise=not mean_only)
        return np.clip(np.asarray(actions), -1., 1.).astype(np.float32)

    @contextmanager
    def evaluation_rng(self, seed):
        policy = self.model.policy
        original = policy.key, policy.noise_key
        policy.key, policy.noise_key = jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 1)
        try:
            yield
        finally:
            policy.key, policy.noise_key = original

    def update(self, step):
        self.model.num_timesteps = step
        # Separate replay sampling on every learner call, as in the source run.
        self.model.train(batch_size=int(self.cfg.alg.batch_size), gradient_steps=1)

    def save(self, path):
        policy = self.model.policy
        state = dict(actor=policy.actor_state, critic=policy.qf_state)
        path.write_bytes(flax.serialization.to_bytes(state))


def evaluate(agent, maze, episodes, seed, output, obstacle=False):
    return evaluate_batched(agent, 'pm_' + maze, seed, episodes, 'mu_only', output, obstacle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maze', choices=tuple(HORIZONS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temperature', type=float, default=5.)
    p.add_argument('--steps', type=int, default=1_000_192)
    p.add_argument('--warmup', type=int, default=8192)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--updates-per-collect', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=200_000)
    p.add_argument('--eval-episodes', type=int, default=200)
    p.add_argument('--final-episodes', type=int, default=500)
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()
    if min(args.num_envs, args.batch_size, args.updates_per_collect, args.eval_every,
           args.eval_episodes, args.final_episodes) <= 0 or args.warmup < args.batch_size:
        p.error('Positive counts and warmup >= batch size are required')
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        p.error('Temperature must be positive and finite')
    if args.eval_episodes % 5 or args.final_episodes % 5:
        p.error('Five-trial robustness requires episode counts divisible by five')
    count = expected_updates(args.steps, args.warmup, args.num_envs, args.updates_per_collect)
    if not args.allow_cpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required (or explicitly use --allow-cpu for tests)')
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = configuration(args.seed, args.temperature, args.batch_size, args.maze)
    cfg.total_steps = args.steps
    metadata = {**vars(args), 'output': str(args.output), 'expected_updates': count,
                'utd': args.updates_per_collect / args.num_envs,
                'learner': OmegaConf.to_container(cfg, resolve=True), 'runtime': provenance(),
                'timeout_bootstrap': False, 'training_conditional_noise': True,
                'evaluation_conditional_noise': False, 'noveld': False}
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2))
    agent = Agent(cfg, args.output)
    environment = TaskBatch(args.maze, args.num_envs, args.seed)
    rng = np.random.default_rng(args.seed)
    try:
        for step in range(args.num_envs, args.steps + 1, args.num_envs):
            obs = environment.current.copy()
            actions = (rng.uniform(-1., 1., (args.num_envs, 2)).astype(np.float32)
                       if step <= args.warmup else agent.act(obs))
            next_obs, rewards, terminated, truncated, _ = environment.step(actions)
            if not np.isfinite(next_obs).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite environment transition')
            done = terminated | truncated
            # Historical PointMaze profile masks BOTH success and time limits.
            agent.replay.add(obs, actions, rewards, next_obs, done.astype(np.float32))
            environment.reset(np.flatnonzero(done))
            if step > args.warmup:
                for _ in range(args.updates_per_collect):
                    agent.update(step)
            if step % args.eval_every < args.num_envs or step == args.steps:
                episodes = args.final_episodes if step == args.steps else args.eval_episodes
                result = dict(primary_evaluation_mode='mu_only')
                result['mu_only'] = evaluate(agent, args.maze, episodes,
                    args.seed + 17000 + step, args.output / f'mu_only_{step}.npz')
                result['obstacle_mu_only'] = evaluate(agent, args.maze, episodes,
                    args.seed + 27000 + step, args.output / f'obstacle_mu_only_{step}.npz', True)
                result.update(steps=step, updates=agent.model._n_updates)
                with (args.output / 'evaluations.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                agent.save(args.output / f'policy_{step}.msgpack')
                agent.model.logger.dump(step)
                print(json.dumps(result), flush=True)
        assert agent.model._n_updates == count
        (args.output / 'completed.json').write_text(json.dumps(dict(steps=step, updates=count)))
    finally:
        environment.close()


if __name__ == '__main__':
    main()
