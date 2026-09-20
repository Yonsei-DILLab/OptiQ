import functools
import time
import tqdm
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as onp
import gymnasium as gym

from src.components.buffer import ReplayBuffer
from src.utils.plotting_utils import fig_to_image, plot_pointmaze_trajectories
from src.agent.dqs import anneal_temperature
from src.agent.critic import get_q_values

import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams.update({
    'figure.figsize': (12, 12),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})


def compute_mmd(x, y, bandwidth=None):
    """Compute unbiased MMD between two point clouds with RBF kernel."""
    x = onp.asarray(x)
    y = onp.asarray(y)
    xx = onp.sum(x**2, axis=1, keepdims=True)
    yy = onp.sum(y**2, axis=1, keepdims=True)
    dxx = xx + xx.T - 2 * x @ x.T
    dyy = yy + yy.T - 2 * y @ y.T
    dxy = xx + yy.T - 2 * x @ y.T
    if bandwidth is None:
        flat = onp.concatenate([dxy.flatten(), dxx.flatten(), dyy.flatten()])
        bandwidth = onp.sqrt(0.5 * onp.median(flat[flat > 0])) + 1e-7
    kxx = onp.exp(-dxx / (2 * bandwidth**2))
    kyy = onp.exp(-dyy / (2 * bandwidth**2))
    kxy = onp.exp(-dxy / (2 * bandwidth**2))
    mmd2 = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return float(onp.sqrt(max(mmd2, 0.0)))


def _extract_obs(observation):
    if isinstance(observation, dict) and 'observation' in observation:
        return observation['observation']
    return observation


class Trainer:
    def __init__(
        self,
        env_fn,
        agent,
        logger,
        num_train_envs=1,
        num_eval_envs=10,
        max_steps=int(1e6),
        replay_buffer_size=int(1e6),
        start_training=int(1e4),
        eval_interval=int(1e5),
        log_interval=int(1000),
        batch_size=512,
        reward_scale=1.0,
        num_updates_per_step=1,
        tqdm_bar=True,
        seed=0,
    ):
        self.env_fn = env_fn
        self.agent = agent
        self.logger = logger

        self.num_train_envs = num_train_envs
        self.num_eval_envs = num_eval_envs
        self.max_steps = max_steps
        self.start_training = start_training
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.num_updates_per_step = num_updates_per_step
        self.reward_scale = reward_scale
        self.tqdm_bar = tqdm_bar
        self.seed = seed

        # Initialize env and buffer spec
        self.env = self.env_fn(seed=self.seed)
        obs0, _ = self.env.reset(seed=self.seed)
        obs0_extracted = _extract_obs(obs0)

        observation_space = getattr(self.env, 'observation_space', None)
        if isinstance(obs0, dict) and hasattr(observation_space, '__getitem__') and 'observation' in observation_space.spaces:
            buffer_obs_space = observation_space['observation']
        else:
            buffer_obs_space = observation_space

        self.replay_buffer = ReplayBuffer(buffer_obs_space, self.env.action_space, replay_buffer_size)

        self.episode_length = int(getattr(self.env.unwrapped, 'max_episode_steps', 1000))

        self.step = 0
        self._last_obs = obs0_extracted

    def _sample_actions(self, observation, diffusion_scale=None):
        out = self.agent.sample_actions(observation, diffusion_scale=diffusion_scale) if diffusion_scale is not None else self.agent.sample_actions(observation)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _mask_function(self, terminated, truncated):
        return 1.0 if (not terminated or truncated) else 0.0

    def train(self):
        observation = self._last_obs
        self.step = 0

        for i in tqdm.tqdm(range(0, self.max_steps, self.num_train_envs), smoothing=0.1, disable=not self.tqdm_bar):
            self.step = i

            if i < self.start_training:
                action = self.env.action_space.sample()
            else:
                action = self._sample_actions(observation)

            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            reward = self.reward_scale * reward

            mask = self._mask_function(terminated, truncated)

            next_observation = _extract_obs(next_obs_raw)
            self.replay_buffer.insert(observation, action, reward, mask, onp.asarray(terminated or truncated, dtype=onp.float32), next_observation)
            observation = next_observation

            if terminated or truncated:
                observation, _ = self.env.reset()
                observation = _extract_obs(observation)

            if i >= self.start_training:
                for _ in range(self.num_updates_per_step):
                    batch = self.replay_buffer.sample(self.batch_size)
                    train_metrics = self.agent.update(*batch)

                if i % self.log_interval == 0:
                    metrics = {'step': self.step}
                    metrics.update(train_metrics)
                    self.logger.log(metrics, 'train')

            if i % self.eval_interval == 0:
                metrics = {'step': self.step}
                eval_metrics = self.evaluate(save_video=False)
                metrics.update(eval_metrics)
                self.logger.log(metrics, 'eval')

        self.step += self.num_train_envs
        metrics = {'step': self.step}
        eval_metrics = self.evaluate(save_video=False)
        metrics.update(eval_metrics)
        self.logger.log(metrics, 'eval')

    def evaluate(self, save_video=False):
        envs = gym.vector.SyncVectorEnv([functools.partial(self.env_fn, seed=i) for i in range(self.num_eval_envs)])
        obs, _ = envs.reset()
        obs = _extract_obs(obs)
        ep_rewards = onp.zeros(self.num_eval_envs)

        for t in range(self.episode_length):
            action = self._sample_actions(obs, diffusion_scale=0.0)
            next_obs_raw, reward, terminated, truncated, info = envs.step(action)
            ep_rewards += onp.array(reward)
            obs = _extract_obs(next_obs_raw)

        return {
            'episode_reward': onp.nanmean(ep_rewards),
        } 


class GMMTrainer(Trainer):
    """Trainer extended with GMM-specific evaluation visualizations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # define action dimension for visualizations
        self.action_dim = self.env.action_space.shape[0]
        # define random key for ground truth sampling
        self.key = jax.random.PRNGKey(self.seed)

    def _mask_function(self, terminated, truncated):
        return 0.0      # for GMM energy is just reward

    def evaluate(self, save_video=False):
        # Reset vectorized envs
        envs = gym.vector.SyncVectorEnv([functools.partial(self.env_fn, seed=i)
                                         for i in range(self.num_eval_envs)])
        obs, _ = envs.reset()
        obs = _extract_obs(obs)
        ep_rewards = onp.zeros(self.num_eval_envs)
        samples = onp.empty((self.num_eval_envs, self.episode_length, self.agent.state_dim))

        for t in range(self.episode_length):
            actions = self._sample_actions(obs, diffusion_scale=0.0)
            next_obs_raw, reward, terminated, truncated, info = envs.step(actions)
            next_obs = _extract_obs(next_obs_raw)
            ep_rewards += onp.array(reward)
            samples[:, t] = next_obs
            obs = next_obs

        # Compute episode reward
        mean_reward = float(onp.nanmean(ep_rewards))
        # Terminal samples slice (-1 is the reset observation, -2 is terminal)
        term_samples = samples[:, -2]
        # Ground truth samples
        gt_samples = self.env.unwrapped.gmm.sample(self.key, term_samples.shape[0])
        # Compute MMD
        mmd_val = compute_mmd(term_samples, gt_samples)

        # Log metrics
        metrics = {'step': self.step, 'episode_reward': mean_reward, 'mmd': mmd_val}
        self.logger.log(metrics, category='eval')

        # Log sample visuals
        img_gen = self.env.unwrapped.get_single_dataset_fig(term_samples, "DQS terminal samples")
        img_true = self.env.unwrapped.get_single_dataset_fig(gt_samples, "Ground truth samples")
        # use logger.log_image to wrap images properly
        self.logger.log_image('true_samples', img_true, self.step, category='plots')
        self.logger.log_image('gen_samples', img_gen, self.step, category='plots')

        # Q-function visualizations
        val_img = self._plot_value_function()
        self.logger.log_image('value_function', val_img, self.step, category='plots')

        # Optionally save video frames as before
        if save_video:
            frames = []
            for t in range(self.episode_length):
                frame = self.env.unwrapped.get_single_dataset_fig(samples[:, t], "frame")
                frames.append(frame)
            video = onp.stack(frames)
            # use logger.video if available
            if hasattr(self.logger, 'video') and self.logger.video:
                self.logger.video.save(self.step)

        return {'episode_reward': mean_reward, 'mmd': mmd_val}

    def _plot_value_function(self, grid_n=120, action_n=41):
        # Grid for states
        plot_bound = self.env.unwrapped.gmm._plot_bound
        bounds = (-plot_bound, plot_bound)
        xs = jnp.linspace(bounds[0], bounds[1], grid_n)
        ys = jnp.linspace(bounds[0], bounds[1], grid_n)
        Sx, Sy = jnp.meshgrid(xs, ys, indexing='xy')
        state_grid = jnp.stack([Sx.ravel(), Sy.ravel()], axis=-1)

        # Grid for actions
        a_lin = jnp.linspace(-1, 1, action_n)
        Ax, Ay = jnp.meshgrid(a_lin, a_lin, indexing='xy')
        action_grid = jnp.stack([Ax.ravel(), Ay.ravel()], axis=-1)

        # Compute V_soft
        alpha = anneal_temperature(onp.log(self.agent.init_temperature),
                                   onp.log(self.agent.final_temperature),
                                   self.step,
                                   self.agent.temperature_steps)
        num_states = state_grid.shape[0]
        num_actions = action_grid.shape[0]
        batch = 512
        v_list = []
        for i in range(0, num_states, batch):
            s_b = state_grid[i:i+batch]
            s_rep = jnp.repeat(s_b[:, None, :], num_actions, axis=1).reshape(-1, self.agent.state_dim)
            a_rep = jnp.repeat(action_grid[None, :, :], s_b.shape[0], axis=0).reshape(-1, self.action_dim)
            q_vals = get_q_values(self.agent.energy_function.Q_module,
                                  self.agent.energy_function.Q_online_params,
                                  s_rep, a_rep).reshape(s_b.shape[0], num_actions)
            v_list.append(alpha * jsp.logsumexp(q_vals / alpha, axis=1))
        v_soft = jnp.concatenate(v_list).reshape(grid_n, grid_n)

        # Plot V_soft
        fig_val, ax_val = plt.subplots(figsize=(12, 12))
        img = ax_val.imshow(onp.asarray(v_soft), origin='lower', extent=[bounds[0], bounds[1], bounds[0], bounds[1]], cmap='viridis')
        fig_val.colorbar(img, ax=ax_val)
        ax_val.set_title('Soft state-value')
        # Overlay true energy contours
        z = onp.asarray(self.env.unwrapped.gmm.log_prob(state_grid)).reshape(grid_n, grid_n)
        ax_val.contour(onp.asarray(Sx), onp.asarray(Sy), z, levels=50, colors='black',
                        linewidths=1.0, alpha=0.4, linestyles='solid')
        val_img = fig_to_image(fig_val)
        plt.close(fig_val)

        return val_img


class PointMazeTrainer(Trainer):
    """Trainer extended with PointMaze-specific evaluation visualizations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maze_map = getattr(self.env.unwrapped.maze, 'maze_map', None)
        if self.maze_map is None:
            raise AttributeError("Environment has no attribute 'maze_map'")

    def evaluate(self, save_video=False):
        # Reset vectorized envs
        envs = gym.vector.SyncVectorEnv([
            functools.partial(self.env_fn, seed=i)
            for i in range(self.num_eval_envs)
        ])
        obs, _ = envs.reset()
        ep_rewards = onp.zeros(self.num_eval_envs)
        trajectories = onp.empty((self.num_eval_envs, self.episode_length, 2))

        for t in range(self.episode_length):
            # sample actions without diffusion noise
            actions = self._sample_actions(obs['observation'], diffusion_scale=0.0)
            next_obs, reward, terminated, truncated, info = envs.step(actions)
            ep_rewards += onp.array(reward)
            trajectories[:, t, :] = next_obs['achieved_goal']
            obs = next_obs

        # Compute metrics
        mean_reward = float(onp.nanmean(ep_rewards))
        self.logger.log({'step': self.step, 'episode_reward': mean_reward}, category='eval')

        # Plot and log trajectories
        traj_img = plot_pointmaze_trajectories(self.maze_map, trajectories)
        self.logger.log_image('trajectories', traj_img, self.step, category='plots')
        return {'episode_reward': mean_reward}
