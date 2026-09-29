"""GMM40 paper figure: archived seed-2 native samples and N=M=256 iBOLT."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PAPER = ROOT / "artifacts/gmm40_paper_ibolt_20260924"
BASE = ROOT / "artifacts/gmm40_5090_queue/results/results"
IBOLT = (ROOT / "artifacts/gmm40_dacer_off_nm_4090/inputs/"
         "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925/results/ibolt_nm256_s2_100k")
sys.path.insert(0, str(PAPER))
import build_figure as common  # noqa: E402

SEED = 2
METHODS = ("SAC", "SQL", "MFPO", "DIPO", "iBOLT")
SAMPLES = 10_000
ALPHA = 0.2
AREA = 2.10  # pt^2
TARGET = BASE / "target/definition.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def check() -> tuple[dict, dict, dict, dict]:
    target = read(TARGET)
    gt = common.bounded_reference(target)
    np.testing.assert_array_equal(gt, np.load(PAPER / "style_v5_full_policy/ground_truth_samples.npy"))
    counts, _ = common.assignments(gt, target)
    threshold = np.maximum(10, 0.1 * counts)
    assert int((counts >= threshold).sum()) == 40
    samples = {"Ground truth": gt}
    rows = {}
    provenance = {"target_sha256": sha(TARGET), "ground_truth_source": str(TARGET.relative_to(ROOT)),
                  "ground_truth_sample_sha256": sha(PAPER / "style_v5_full_policy/ground_truth_samples.npy"),
                  "seed": SEED, "actor_updates": 100000, "samples_per_panel": SAMPLES,
                  "point_alpha": ALPHA, "point_area_pt2": AREA, "point_color": "#0000FF",
                  "plotter_sha256": sha(Path(__file__))}
    for method in METHODS:
        run = IBOLT if method == "iBOLT" else BASE / f"{method.lower()}_s{SEED}_100k"
        cfg, audit = read(run / "config.json"), read(run / "update_count_audit.json")
        assert cfg["seed"] == SEED and cfg["steps"] == 100000 and cfg["batch"] == 256
        assert audit["status"] == "passed" and audit["actor_updates"] == 100000
        assert audit["full_budget_completed"] and not audit["errors"]
        if method == "iBOLT":
            assert cfg["n"] == cfg["m"] == 256 and (cfg["width"], cfg["depth"]) == (256, 3)
            assert cfg["latent_mode"] == "random"
            receipt = read(IBOLT.parents[1] / "LOCAL_COPY_VERIFIED.json")
            assert receipt["campaign"] == "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925"
        other_target = read(run.parent / "target/definition.json")
        for key in ("means", "std", "weights", "scale", "bounded_mass"):
            np.testing.assert_array_equal(other_target[key], target[key])
        eval_dir = run / "evaluations/step_0100000"
        sample_file, metric_file = eval_dir / "samples.npy", eval_dir / "metrics.json"
        x, metric = np.load(sample_file, allow_pickle=False), read(metric_file)
        assert x.shape == (SAMPLES, 2) and np.isfinite(x).all() and (np.abs(x) <= 40.0001).all()
        n, near = common.assignments(x, target)
        np.testing.assert_array_equal(n, metric["mode_counts_3sigma"])
        np.testing.assert_allclose(threshold, metric["coverage_threshold"], atol=1e-12)
        assert int((n >= threshold).sum()) == metric["mode_coverage"]
        assert abs(near.mean() - metric["high_density_fraction"]) < 1e-12
        mmd2 = common.verify_mmd2(x, gt)
        assert abs(mmd2 - metric["mmd2"]) < 1e-10
        assert mmd2 > 0
        rows[method] = {"mmd": math.sqrt(mmd2), "mmd2": mmd2,
                        "mode_coverage": int(metric["mode_coverage"]),
                        "near_fraction": float(metric["high_density_fraction"]),
                        "source_commit": cfg["source_git_commit"]}
        samples[method] = x
        provenance[method] = {"evaluation_mode": "full_policy_random_latent_conditional_sigma"
                              if method == "iBOLT" else "native_policy_generator",
                              "run": str(run.relative_to(ROOT)), "sample_sha256": sha(sample_file),
                              "metric_sha256": sha(metric_file), "config_sha256": sha(run / "config.json"),
                              "audit_sha256": sha(run / "update_count_audit.json")}
    return target, samples, rows, provenance


def panel(ax, x: np.ndarray, grid, title: str, *, ylabel: bool = True) -> None:
    common.panel(ax, x, grid, title, ylabel=ylabel, size=AREA)
    dots = ax.collections[-1]
    dots.set_facecolor("#0000FF")
    dots.set_alpha(ALPHA)
    dots.set_rasterized(True)
    np.testing.assert_array_equal(dots.get_offsets(), x)
    np.testing.assert_array_equal(dots.get_sizes(), [AREA])
    np.testing.assert_allclose(dots.get_facecolors(), [[0, 0, 1, ALPHA]], atol=1e-15)


def render(target: dict, samples: dict) -> None:
    common.style()
    grid = common.contour_grid(target)
    methods = ("Ground truth",) + METHODS
    for layout, figsize in (("main", (9.0, 6.45)), ("wide", (16.2, 3.15))):
        if layout == "main":
            fig, axs = plt.subplots(2, 3, figsize=figsize)
            axes = axs.flat
        else:
            fig, axes = plt.subplots(1, 6, figsize=figsize)
        for i, (ax, name) in enumerate(zip(axes, methods)):
            title = f"({chr(97+i)}) {name}" + (" (ours)" if name == "iBOLT" else "")
            panel(ax, samples[name], grid, title, ylabel=(i % 3 == 0 if layout == "main" else i == 0))
        if layout == "main":
            fig.subplots_adjust(left=.055, right=.99, bottom=.07, top=.94, hspace=.30, wspace=.20)
        else:
            fig.subplots_adjust(left=.028, right=.995, bottom=.22, top=.86, wspace=.28)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"gmm40_seed2_nm256_{layout}.{ext}", dpi=360,
                        bbox_inches="tight", pad_inches=.045)
        plt.close(fig)


def table(rows: dict) -> str:
    names = ["Ground truth", "SAC", "SQL", "MFPO", "DIPO", r"\textbf{iBOLT (ours)}"]
    mmd = ["--"] + [f"{rows[n]['mmd']:.4f}" for n in METHODS[:-1]] + [rf"\textbf{{{rows['iBOLT']['mmd']:.4f}}}"]
    coverage = ["40/40"] + [f"{rows[n]['mode_coverage']}/40" for n in METHODS[:-1]] + [rf"\textbf{{{rows['iBOLT']['mode_coverage']}/40}}"]
    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{GMM40 sampling quality after 100{,}000 actor updates (training seed 2). "
        r"Each method contributes 10{,}000 final-policy samples. MMD uses the first "
        r"2{,}048 samples and the same independent ground-truth draw; ground truth "
        r"is the reference, so its MMD entry is omitted.}",
        r"\label{tab:gmm40_seed2_nm256}",
        r"\footnotesize",
        r"\resizebox{0.85\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        "    Algorithm & " + " & ".join(names) + r" \\",
        r"    \midrule",
        "    MMD $\\downarrow$ & " + " & ".join(mmd) + r" \\",
        "    Mode coverage $\\uparrow$ & " + " & ".join(coverage) + r" \\",
        r"    \bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.2em}",
        r"\end{table}",
        "",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target, samples, rows, provenance = check()
    render(target, samples)
    (OUT / "gmm40_seed2_nm256_table.tex").write_text(table(rows))
    (OUT / "results.json").write_text(json.dumps({"rows": rows, "provenance": provenance}, indent=2) + "\n")
    (OUT / "figure_caption.txt").write_text(
        "GMM40 final-policy samples at 100,000 actor updates (seed 2). "
        "Panels show ground truth, SAC, SQL, MFPO, DIPO, and iBOLT (N=M=256). "
        "Each panel contains 10,000 samples. iBOLT samples include random latent "
        "and learned conditional Gaussian noise; other methods use their native "
        "policy sampler. Density contours show the same ground-truth distribution "
        "in every panel. Blue markers use alpha 0.2 and area 2.10 pt^2.\n")
    print(json.dumps({"rows": rows, "figure": str(OUT / "gmm40_seed2_nm256_main.png")}, indent=2))


if __name__ == "__main__":
    main()
