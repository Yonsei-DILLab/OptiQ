"""Paper-style four-way policy arrows and learned value surfaces.

Top: the SAME goal-proximity contour for every method, plus stochastic policy
action arrows conditioned on each (x, y) state. The contour is reference task
geometry, not an estimated policy density. Bottom: each method's own
E_a[Q(s,a)] on the same wall-free grid. The 3D surfaces are learned values,
not oracle returns, and different algorithms' critic objectives can differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GOAL_LABELS = ("E", "W", "N", "S")
GOAL_XY = np.array([[5., 0.], [-5., 0.], [0., 5.], [0., -5.]])


def probe_policy(agent, destination: Path, seed=104729, samples=8, chunk=128, mode=None):
    """Save policy/Q evaluations over the wall-free 2D state plane."""
    coordinates = np.linspace(-7., 7., 29)
    if mode is None:
        mode = "mu_only" if getattr(agent, "method", None) == "optiq" else "policy"
    if getattr(agent, "method", None) == "optiq" and mode != "mu_only":
        raise ValueError("OptiQ probes require mu_only")
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    valid = np.ones(len(xy), bool)
    states = xy.astype(np.float32)
    repeated = np.repeat(states.astype(np.float32), samples, axis=0)
    actions = np.empty((len(repeated), 2), np.float32)
    values = np.empty(len(repeated), np.float32)
    with agent.evaluation_rng(seed):
        for left in range(0, len(repeated), chunk):
            right = min(left + chunk, len(repeated))
            actions[left:right] = agent.act(repeated[left:right], mode=mode)
            values[left:right] = agent.q(repeated[left:right], actions[left:right])
    if not np.isfinite(actions).all() or not np.isfinite(values).all():
        raise FloatingPointError("nonfinite policy/Q grid probe")
    action_grid = np.full(xx.shape + (samples, 2), np.nan, np.float32)
    q_grid = np.full(xx.shape + (samples,), np.nan, np.float32)
    action_grid.reshape(-1, samples, 2)[valid] = actions.reshape(-1, samples, 2)
    q_grid.reshape(-1, samples)[valid] = values.reshape(-1, samples)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=coordinates, y=coordinates, valid=valid.reshape(xx.shape),
                        actions=action_grid, q=q_grid,
                        samples=samples, critic_label=agent.critic_label, evaluation_mode=mode)
    return destination


def _rollouts(path: Path):
    with np.load(path) as data:
        if "xy" in data:
            paths = data["xy"]
        elif "trajectories" in data:
            paths = data["trajectories"]
        else:
            raise KeyError("rollout file needs xy or trajectories")
        if "goal_ids" in data:
            goals = data["goal_ids"]
        elif "goals" in data:
            goals = data["goals"]
        else:
            raise KeyError("rollout file needs goal_ids or goals")
    if paths.ndim != 3 or paths.shape[-1] != 2 or goals.shape != (len(paths),):
        raise ValueError("invalid rollout dimensions")
    return paths, goals


def _goal_proximity():
    """DACER's four-Gaussian display field, not a learned policy distribution.

    Source: happy-yan/DACER-Diffusion-with-Online-RL, commit 9f22f29,
    relax_env/multigoal.py::_plot_position_cost. The figure uses sigma=1.7,
    amplitude 40/(2*pi*sigma**2), and sums the four goal-centered Gaussians.
    """
    axis = np.linspace(-7.7, 7.7, 309)
    xx, yy = np.meshgrid(axis, axis)
    sigma = 1.7
    amplitude = 40. / (2. * np.pi * sigma ** 2)
    field = np.sum([
        amplitude * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2. * sigma ** 2))
        for gx, gy in GOAL_XY
    ], axis=0)
    return xx, yy, field


def _visited_cells(paths, x, y):
    """Select actual visited grid cells instead of arrows at unreachable states."""
    xy = paths.reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    dx, dy = x[1] - x[0], y[1] - y[0]
    col = np.rint((xy[:, 0] - x[0]) / dx).astype(int)
    row = np.rint((xy[:, 1] - y[0]) / dy).astype(int)
    inside = (row >= 0) & (row < len(y)) & (col >= 0) & (col < len(x))
    counts = np.zeros((len(y), len(x)), np.int32)
    np.add.at(counts, (row[inside], col[inside]), 1)
    selected = (counts >= 2) & ((np.indices(counts.shape).sum(axis=0) % 2) == 0)
    return selected


def render(panels: list[tuple[str, Path, Path]], output: Path):
    """Render one or more methods in the supplied order."""
    loaded = []
    for name, rollout_path, probe_path in panels:
        paths, goals = _rollouts(rollout_path)
        with np.load(probe_path) as probe:
            x, y, valid = probe["x"], probe["y"], probe["valid"]
            actions, q = probe["actions"], probe["q"]
            label = str(probe["critic_label"])
            mode = str(probe["evaluation_mode"]) if "evaluation_mode" in probe else "historical full policy"
            label += f"; action mode: {mode}"
        if actions.shape[:2] != valid.shape or q.shape[:2] != valid.shape:
            raise ValueError(f"{name}: probe shape mismatch")
        counts = np.bincount(goals[goals >= 0].astype(int), minlength=4)
        visited = _visited_cells(paths, x, y) & valid
        loaded.append((name, x, y, valid, actions, q, visited, counts,
                       int((goals < 0).sum()), label))

    plt.rcParams.update({"font.family": "DejaVu Serif"})
    figure = plt.figure(figsize=(4.25 * len(loaded), 8.7), constrained_layout=False)
    layout = figure.add_gridspec(2, len(loaded), left=.055, right=.985,
                                 top=.95, bottom=.145, wspace=.20, hspace=.13,
                                 height_ratios=(1., 1.05))
    contour_x, contour_y, proximity = _goal_proximity()
    for col, (name, x, y, valid, actions, q, visited, counts, failures, label) in enumerate(loaded):
        top = figure.add_subplot(layout[0, col])
        xx, yy = np.meshgrid(x, y)
        levels = np.linspace(float(np.min(proximity)), float(np.max(proximity)), 20)
        contours = top.contour(contour_x, contour_y, proximity, levels=levels,
                               cmap="viridis", linewidths=.72)
        top.clabel(contours, contours.levels[::2], inline=True, fontsize=5.7, fmt="%.1f")
        # One independent policy draw at each actually visited cell; multiple
        # draws at a single location would obscure the arrow directions.
        rows, cols = np.nonzero(visited)
        sample_ids = (rows * 131 + cols * 17) % actions.shape[-2]
        selected_actions = actions[rows, cols, sample_ids]
        top.quiver(x[cols], y[rows], selected_actions[:, 0] * 3., selected_actions[:, 1] * 3.,
                   color="red", alpha=.9, angles="xy", scale_units="xy",
                   scale=2., width=.005, headwidth=3.7, headlength=5.)
        # At the shared fixed start, display independent policy draws around
        # a small circle to expose any locally multimodal choices.
        center_row, center_col = len(y) // 2, len(x) // 2
        origins = np.array([[.19, .19], [-.19, .19], [-.19, -.19], [.19, -.19]])
        origin_actions = actions[center_row, center_col, :4]
        top.quiver(origins[:, 0], origins[:, 1], origin_actions[:, 0] * 3., origin_actions[:, 1] * 3.,
                   color="red", alpha=.9, angles="xy", scale_units="xy",
                   scale=2., width=.005, headwidth=3.7, headlength=5.)
        for gx, gy in GOAL_XY:
            top.scatter(gx, gy, s=24, color="red", zorder=5)
        top.set(xlim=(-7.7, 7.7), ylim=(-7.7, 7.7), aspect="equal")
        top.tick_params(labelsize=7, length=2)

        bottom = figure.add_subplot(layout[1, col], projection="3d")
        value = np.full(valid.shape, np.nan, np.float32)
        value[valid] = q[valid].mean(axis=-1)
        bottom.plot_surface(xx, yy, np.ma.masked_where(~valid, value),
                            cmap="viridis", linewidth=0, antialiased=True)
        bottom.set(xlabel="x", ylabel="y", zlabel="Value function")
        bottom.view_init(elev=28, azim=-58)
        bottom.tick_params(labelsize=6)
        bottom.set_box_aspect((1, 1, .65))
        letter = chr(ord("a") + col)
        center = (col + .5) / len(loaded)
        figure.text(center, .127, f"({letter}) {name}", ha="center", fontsize=14)
        figure.text(center, .102,
                    f"E {counts[0]}   W {counts[1]}   N {counts[2]}   S {counts[3]}   fail {failures}",
                    ha="center", fontsize=8)
        figure.text(center, .083, label, ha="center", fontsize=6.8)
    figure.text(.055, .035,
                "DACER four-Gaussian contours; red arrows: sampled policy actions.\n"
                "Contours are not policy density. Bottom: learned Q averaged over the indicated action sampler; not oracle V.",
                fontsize=7.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=175)
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", nargs=3, action="append", metavar=("METHOD", "ROLLOUT", "PROBE"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render([(name, Path(rollout), Path(probe)) for name, rollout, probe in args.panel], args.output)


if __name__ == "__main__":
    main()
