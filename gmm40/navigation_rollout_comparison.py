"""Compare saved environment trajectories at one common training checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .evaluation import atomic_json, background
from .target import RESULTS, Target


RUNS = [
    ("OptiQ T=0.25", "navigation_optiq_T025_seed0_100k"),
    ("SAC (auto alpha)", "navigation_sac_seed0_100k"),
    ("DIPO", "navigation_dipo_native_seed0_100k"),
    ("MEow (alpha=1)", "navigation_meow_seed0_100k"),
    ("MFPO (auto alpha)", "navigation_mfpo_native_seed0_100k"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(updates):
    evaluations, inputs, missing = {}, [], []
    initial = None
    for label, name in RUNS:
        directory = RESULTS / name / "evaluations" / f"update_{updates:07d}"
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            missing.append(name)
            continue
        metrics = json.loads(metrics_path.read_text())
        rollout_path = directory / "environment_rollout.npz"
        with np.load(rollout_path) as data:
            positions = data["positions"].copy()
            rewards = data["rewards"].copy()
        assert metrics["updates"] == updates, name
        assert positions.shape == (101, 1000, 2), (name, positions.shape)
        assert rewards.shape == (100, 1000), (name, rewards.shape)
        assert np.isfinite(positions).all() and np.isfinite(rewards).all(), name
        assert np.isclose(rewards.sum(0).mean(), metrics["mean_return"]), name
        if initial is None:
            initial = positions[0].copy()
        else:
            assert np.array_equal(initial, positions[0]), f"Unpaired starts: {name}"
        evaluations[name] = (positions, metrics)
        inputs.append(dict(run=name, updates=updates,
                           rollout=str(rollout_path.relative_to(RESULTS)),
                           rollout_sha256=digest(rollout_path),
                           metrics=str(metrics_path.relative_to(RESULTS)),
                           metrics_sha256=digest(metrics_path)))
    if not evaluations:
        raise ValueError(f"No published evaluations at {updates} updates")

    target = Target()
    reference = target.sample(1000, 20260917, bounded=False)
    times = list(range(0, 101, 4))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.8), dpi=100)
    fig.subplots_adjust(top=.83, bottom=.08, hspace=.43, wspace=.18)
    for ax in axes.flat:
        background(ax, target)
        ax.set(xlim=(-50, 50), ylim=(-50, 50))
    axes.flat[0].scatter(*reference.T, s=4, alpha=.4, color="#3676b9")
    axes.flat[0].set_title("Ground truth: original GMM40\nFixed reference; not a trajectory", fontsize=11)
    artists = {}
    for ax, (label, name) in zip(list(axes.flat)[1:], RUNS):
        if name not in evaluations:
            ax.set_title(f"{label}\n{updates:,}-update evaluation not yet available", fontsize=11)
            continue
        metric = evaluations[name][1]
        ax.set_title(f"{label} | final return {metric['mean_return']:.1f}\n"
                     f"Final MMD² {metric['mmd2']:.4f} | {metric['mode_coverage']}/40 components", fontsize=11)
        trails = [ax.plot([], [], lw=.7, alpha=.45, color="#98501f")[0] for _ in range(16)]
        points = ax.scatter([], [], s=4, alpha=.4, color="#dd7932")
        artists[name] = (points, trails)
    heading = fig.suptitle("", fontsize=15, y=.975)
    fig.text(.5, .90, "Orange: 1,000 policy positions | Brown lines: first 16 paths | Blue: target samples | Gray contours / black crosses: target",
             ha="center", fontsize=10)
    fig.text(.5, .025, "Same initial positions; stochastic policies. Animation time is environment time, not learning progress. Panel metrics always describe step 100.",
             ha="center", fontsize=10)
    frames = []
    for t in times:
        heading.set_text(f"Moving-Q policies at {updates:,} actor updates | environment step {t}/100\n"
                         f"Seed 0 | {len(evaluations)}/5 methods available at this checkpoint")
        for name, (points, trails) in artists.items():
            positions = evaluations[name][0]
            points.set_offsets(positions[t])
            for i, line in enumerate(trails):
                line.set_data(positions[:t + 1, i, 0], positions[:t + 1, i, 1])
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    stem = f"navigation_rollout_comparison_{updates:07d}"
    output = RESULTS / f"{stem}.gif"
    durations = [180] * (len(frames) - 1) + [2500]
    frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, disposal=2)
    fig.savefig(RESULTS / f"{stem}_final.png", dpi=120)
    fig.savefig(RESULTS / f"{stem}_final.pdf")
    plt.close(fig)
    with Image.open(output) as animation:
        assert animation.n_frames == len(times)
        for index, expected in enumerate(durations):
            animation.seek(index)
            animation.load()
            assert animation.info["duration"] == expected
        dimensions = list(animation.size)
    atomic_json(RESULTS / f"{stem}_manifest.json", dict(
        kind="actual 100-step environment trajectories at matched training updates",
        updates=updates, environment_steps=times, frames=len(frames), dimensions=dimensions,
        episodes=1000, paths_shown=16, reset_seed=20260918, training_seed=0,
        full_stochastic_policy=True, input_initial_positions_exactly_equal=True,
        panel_metrics="saved terminal metrics; not recomputed at intermediate environment times",
        available_methods=len(evaluations), missing_runs=missing,
        target_sha256=digest(RESULTS / "target" / "definition.json"),
        renderer_sha256=digest(Path(__file__)), output_sha256=digest(output), inputs=inputs,
        validation="matched updates; finite arrays; paired starts; return agrees; all GIF frames decode with positive specified delays"))
    print(json.dumps(dict(output=str(output), frames=len(frames), available=len(evaluations), missing=missing)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100000)
    render(parser.parse_args().updates)
