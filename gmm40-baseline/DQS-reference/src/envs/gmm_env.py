from typing import Optional
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
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

import jax
import jax.numpy as jnp
import distrax
try:
    import wandb
except Exception:
    wandb = None

from src.utils.plotting_utils import plot_contours, plot_marginal_pair, fig_to_image

import gymnasium as gym
from gymnasium import spaces


class GMM:
    def __init__(
        self,
        dim: int = 2, n_mixes: int = 40, loc_scaling: float = 40,
        scale_scaling: float = 1.0, seed: int = 1,
    ):

        self.seed = seed
        self.n_mixes = n_mixes

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        # GMM experiment in the paper
        mean = jnp.array([[ -0.2995,  21.4577],
            [-32.9218, -29.4376],
            [-15.4062,  10.7263],
            [ -0.7925,  31.7156],
            [ -3.5498,  10.5845],
            [-12.0885,  -7.8626],
            [-38.2139, -26.4913],
            [-16.4889,   1.4817],
            [ 15.8134,  24.0009],
            [-27.1176, -17.4185],
            [ 14.5287,  33.2155],
            [ -8.2320,  29.9325],
            [ -6.4473,   4.2326],
            [ 36.2190, -37.1068],
            [-25.1815, -10.1266],
            [-15.5920,  34.5600],
            [-25.9272, -18.4133],
            [-27.9456, -37.4624],
            [-23.3496,  34.3839],
            [ 17.8487,  19.3869],
            [  2.1037, -20.5073],
            [  6.7674, -37.3478],
            [-28.9026, -20.6212],
            [ 25.2375,  23.4529],
            [-17.7398,  -1.4433],
            [ 25.5824,  39.7653],
            [ 15.8753,   5.4037],
            [ 26.8195, -23.5521],
            [  7.4538, -31.0122],
            [-27.7234, -20.6633],
            [ 18.0989,  16.0864],
            [-23.6941,  12.0843],
            [ 21.9589,  -5.0487],
            [  1.5273,   9.2682],
            [ 24.8151,  38.4078],
            [-30.8249, -14.6588],
            [ 15.7204,  33.1420],
            [ 34.8083,  35.2943],
            [  7.9606, -34.7833],
            [  3.6797, -25.0242]])
        scale = jnp.ones(shape=(n_mixes, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )
        self._plot_bound = loc_scaling * 1.2

    @partial(jax.jit, static_argnums=(0,))
    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, seed, sample_shape):
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)


class GMMEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super(GMMEnv, self).__init__()

        self.gmm = GMM(
            dim=2,
            n_mixes=40,
            loc_scaling=40,
            scale_scaling=1.0,
            seed=0,
        )

        self.observation_space = spaces.Box(low=np.array([-50.0, -50.0], dtype=np.float32), high=np.array([50.0, 50.0], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([-50.0, -50.0]), high=np.array([50.0, 50.0]), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.max_episode_steps = 100
        self.num_steps = None
        self.state = None

        self.path = []
        self.render_mode = render_mode
    
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        self.num_steps += 1

        # normalize action to be unit vector
        norm = np.linalg.norm(action, axis=-1, keepdims=True) + 1e-8
        action = action / norm
        delta_x = action[0]
        delta_y = action[1]
        new_state = self.state + np.array([delta_x, delta_y])

        # Clip the new state within the state space boundaries
        self.state = np.clip(new_state, self.observation_space.low, self.observation_space.high)

        # Reward is log prob of state under GMM
        reward = float(self.gmm.log_prob(self.state))

        # Termination/truncation
        terminated = self.num_steps >= self.max_episode_steps
        truncated = False

        self.path.append(self.state)

        info = {}

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.num_steps = 0
        # Reset the agent's state to a random position within the state space
        self.state = np.random.uniform(self.observation_space.low, self.observation_space.high).astype(np.float32)

        self.path = [self.state]

        info = {}
        return np.array(self.state, dtype=np.float32), info

    def render(self):
        if self.render_mode == 'human':
            plt.figure(figsize=(12, 12), dpi=300)
            plt.plot([p[0] for p in self.path], [p[1] for p in self.path], marker='o', linestyle='-', color='b')
            plt.title("Agent's Path")
            plt.xlim(self.observation_space.low[0], self.observation_space.high[0])
            plt.ylim(self.observation_space.low[1], self.observation_space.high[1])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig('nav2d.png')
            plt.close()

    def log_on_epoch_end(
        self,
        latest_samples,
        latest_energies,
        wandb_logger,
        prefix="",
        step=0,
    ) -> None:
        if wandb_logger is None or wandb is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if latest_samples is not None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.scatter(*latest_samples.T)

            wandb_logger.log({f"{prefix}generated_samples_scatter": wandb.Image(fig_to_image(fig)), "step": step}, img=True)
            img = self.get_single_dataset_fig(latest_samples, "dqs_generated_samples")
            wandb_logger.log({f"{prefix}generated_samples": wandb.Image(img), "step": step}, img=True)

        plt.close('all')

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.2 * 40, 1.2 * 40)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            levels=50,
        )

        plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-1.2 * 40, 1.2 * 40)):
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))

        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            levels=50,
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                levels=50,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        return fig_to_image(fig)

    def close(self):
        pass