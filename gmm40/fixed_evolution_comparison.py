"""Render saved fixed-Q evaluations at matched updates; no learner execution."""
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
    ("OptiQ 16 x 64", "optiq_n16_m64_seed0_100k"),
    ("SAC", "sac_seed0_100k"),
    ("DIPO", "dipo_native_seed0_100k"),
    ("MEow", "meow_seed0"),
    ("MFPO", "mfpo_native_seed0"),
]
STEPS = [0, 100, 500, 1000, 2500, 5000, *range(10000, 100001, 10000)]


def evaluation_paths(folder):
    config = json.loads((folder / "config.json").read_text())
    paths = {}
    if config.get("resume"):
        parent = Path(config["resume"]).parent.parent
        if parent.resolve() == folder.resolve():
            raise ValueError("Self-referential resume path")
        paths.update(evaluation_paths(parent))
    for line in (folder / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            step = json.loads(line)["step"]
            paths[step] = folder / "evaluations" / f"step_{step:07d}"
    return paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render():
    target = Target()
    reference = target.sample(10000, 20260917)
    inputs = []
    evaluations = []
    for label, name in RUNS:
        folder = RESULTS / name
        assert json.loads((folder / "status.json").read_text())["status"] == "completed"
        paths = evaluation_paths(folder)
        assert all(step in paths for step in STEPS), (name, "Missing matched evaluation")
        results = {}
        for step in STEPS:
            samples_path = paths[step] / "samples.npy"
            metrics_path = paths[step] / "metrics.json"
            samples = np.load(samples_path)
            metrics = json.loads(metrics_path.read_text())
            assert samples.shape == (10000, 2) and np.isfinite(samples).all()
            assert metrics["step"] == step
            results[step] = (samples[:5000], metrics)
            inputs.append(dict(run=name, updates=step,
                               samples=str(samples_path.relative_to(RESULTS)),
                               samples_sha256=digest(samples_path),
                               metrics=str(metrics_path.relative_to(RESULTS)),
                               metrics_sha256=digest(metrics_path)))
        evaluations.append((label, results))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), dpi=100)
    fig.subplots_adjust(top=.84, bottom=.08, hspace=.37, wspace=.16)
    for ax in axes.flat:
        background(ax, target)
    axes.flat[0].scatter(*reference[:5000].T, s=2, alpha=.35, color="#3676b9")
    axes.flat[0].set_title("Ground truth: bounded GMM40\nSame reference in every frame", fontsize=11)
    plots = [ax.scatter([], [], s=2, alpha=.35, color="#dd7932") for ax in list(axes.flat)[1:]]
    heading = fig.suptitle("", fontsize=15, y=.97)
    fig.text(.5, .908, "Blue: target samples | Orange: full stochastic policy | Gray contours / black crosses: target density / component centers",
             ha="center", fontsize=10)
    fig.text(.5, .025, "Training checkpoints, not sampler time or environment steps. Frames are not uniformly spaced in updates. Seed 0; one training run per method.",
             ha="center", fontsize=10)
    frames = []
    for step in STEPS:
        heading.set_text(f"Fixed Q = log p_GMM40 | {step:,} actor updates\nT = 1 for temperature-based methods; DIPO has no entropy objective")
        for ax, plot, (label, results) in zip(list(axes.flat)[1:], plots, evaluations):
            samples, metric = results[step]
            plot.set_offsets(samples)
            ax.set_title(f"{label} | {metric['mode_coverage']}/40 components\n"
                         f"MMD² {metric['mmd2']:.4f} | within 3σ: {metric['high_density_fraction']:.1%}", fontsize=11)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    output = RESULTS / "fixed_training_comparison.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:], loop=0,
                   duration=[1300] * (len(frames) - 1) + [3500], disposal=2)
    fig.savefig(RESULTS / "fixed_training_comparison_final.png", dpi=120)
    fig.savefig(RESULTS / "fixed_training_comparison_final.pdf")
    plt.close(fig)
    with Image.open(output) as animation:
        assert animation.n_frames == len(STEPS)
        for frame in range(animation.n_frames):
            animation.seek(frame)
            animation.load()
        dimensions = list(animation.size)
    atomic_json(RESULTS / "fixed_training_comparison_manifest.json", dict(
        kind="matched-update fixed-Q learning progression; not generation rollout",
        updates=STEPS, frames=len(frames), dimensions=dimensions,
        points_shown_per_panel=5000, metrics="saved primary evaluations; not recomputed on plot subset",
        reference_seed=20260917, reference_total_samples=10000,
        full_stochastic_policy=True, seed=0,
        target_sha256=digest(RESULTS / "target" / "definition.json"),
        renderer_sha256=digest(Path(__file__)),
        output_sha256=digest(output), inputs=inputs,
        validation="all requested matched checkpoints exist; finite 10000x2 arrays; GIF frames decode"))
    print(json.dumps(dict(output=str(output), frames=len(frames), dimensions=dimensions,
                          source_evaluations=len(inputs))))


if __name__ == "__main__":
    render()
