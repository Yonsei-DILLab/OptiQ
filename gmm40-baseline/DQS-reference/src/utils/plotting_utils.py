import itertools
import PIL
from matplotlib import pyplot as plt
import jax.numpy as jnp
import numpy as onp
from matplotlib import patches


def plot_contours(log_prob_func,
                     ax=None,
                     bounds=(-5, 5),
                     levels=20):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = onp.linspace(bounds[0], bounds[1], n_points)
    x_points_dim2 = onp.linspace(bounds[0], bounds[1], n_points)
    x_points = onp.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels, linewidths=2.5)


def plot_marginal_pair(samples,
                  ax=None,
                  marginal_dims=(0, 1),
                  bounds=(-5, 5),
                  alpha=0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)


def fig_to_image(fig):
    """Convert a Matplotlib figure to a PIL Image.
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = onp.frombuffer(fig.canvas.buffer_rgba(), dtype=onp.uint8)
    rgba = rgba.reshape((height, width, 4))
    return PIL.Image.fromarray(rgba, mode="RGBA").convert("RGB")


def plot_pointmaze_trajectories(maze_map, trajectories, figsize=(12, 12), dpi=300):
    """
    Plot trajectories over the env.maze_map and return as Image.
    trajectories: array of shape (N, T, 2)
    env: a gym env instance with attribute maze_map
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # scatter trajectories
    for traj in trajectories:
        pts = onp.array(traj)
        x, y = pts[:, 0], pts[:, 1]
        ax.scatter(x, y, s=10, c='C1', alpha=0.3)
        ax.scatter(x[-1], y[-1], s=100, c='C1', marker='*', alpha=0.2)
    # draw maze walls
    S = 1
    cols = len(maze_map[0])
    rows = len(maze_map)
    x_offset = cols / 2 * S
    y_offset = rows / 2 * S
    for i in range(rows):
        for j in range(cols):
            if maze_map[i][j] == 1:
                xpos = (j + 0.5) * S - x_offset - S/2
                ypos = y_offset - (i + 0.5) * S - S/2
                rect = patches.Rectangle((xpos, ypos), S, S,
                                         linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)
                ax.add_patch(rect)
            elif maze_map[i][j] == 'g':
                x_center = (j + 0.5) * S - x_offset
                y_center = y_offset - (i + 0.5) * S
                ax.scatter(x_center, y_center, s=1600, c='C1', marker='*')
    ax.set_xlim(-x_offset, cols * S - x_offset)
    ax.set_ylim(y_offset - rows * S, y_offset)
    ax.set_title('Trajectories', fontsize=22)
    plt.tight_layout()
    img = fig_to_image(fig)
    plt.close(fig)
    return img
