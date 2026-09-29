"""GMM40 paper figure: seed-2 panels, four-seed mean table, SQL K=256."""
from __future__ import annotations

import hashlib
import json
import math
import csv
import shutil
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PAPER = ROOT / "artifacts/gmm40_paper_ibolt_20260924"
BASE = ROOT / "artifacts/gmm40_5090_queue/results/results"
IBOLT_ROOT = (ROOT / "artifacts/gmm40_dacer_off_nm_4090/inputs/"
              "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925/results")
SQL_ROOT = ROOT / "artifacts/gmm40_sql_particles_100k"
sys.path.insert(0, str(PAPER))
import build_figure as common  # noqa: E402

SEED = 2
METHODS = ("SAC", "SQL", "MFPO", "DIPO", "iBOLT")
SAMPLES = 10_000
ALPHA = 0.2
AREA = 2.10  # pt^2
TARGET = BASE / "target/definition.json"


def run_for(method: str, seed: int) -> Path:
    if method == "SQL":
        return SQL_ROOT / f"shard{seed//2}/results/sql_k256_s{seed}_100k"
    if method == "iBOLT":
        return IBOLT_ROOT / f"ibolt_nm256_s{seed}_100k"
    return BASE / f"{method.lower()}_s{seed}_100k"


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
        run = run_for(method, SEED)
        cfg, audit = read(run / "config.json"), read(run / "update_count_audit.json")
        assert cfg["seed"] == SEED and cfg["steps"] == 100000 and cfg["batch"] == 256
        assert audit["status"] == "passed" and audit["actor_updates"] == 100000
        assert audit["full_budget_completed"] and not audit["errors"]
        if method == "iBOLT":
            assert cfg["n"] == cfg["m"] == 256 and (cfg["width"], cfg["depth"]) == (256, 3)
            assert cfg["latent_mode"] == "random"
            receipt = read(IBOLT_ROOT.parent / "LOCAL_COPY_VERIFIED.json")
            assert receipt["campaign"] == "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925"
        if method == "SQL":
            assert cfg["sql_kernel_particles"] == 256 and cfg["sql_kernel_update_ratio"] == .5
            assert cfg["sql_value_particles"] == 16 and (cfg["width"], cfg["depth"]) == (256, 2)
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
    xx, yy, relative, levels, _ = grid
    # Keep the original density and contour locations; only increase color contrast.
    cmap = LinearSegmentedColormap.from_list("gmm40_density_contrast", [
        (0.00, "#080012"), (0.30, "#19002f"), (0.35, "#304ab0"),
        (0.68, "#1097b7"),
        (0.88, "#4dbb73"), (1.00, "#e4cf2d")])
    ax.contour(xx, yy, relative, levels=levels, cmap=cmap,
               norm=Normalize(vmin=-140, vmax=0, clip=True),
               linewidths=.34, alpha=.95, zorder=1)
    ax.scatter(x[:, 0], x[:, 1], s=AREA, c="#0000FF", alpha=ALPHA,
               marker="o", linewidths=0, rasterized=True, zorder=2)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal")
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(pad=2)
    ax.set_title(title, pad=8, fontweight="bold" if "iBOLT" in title else "normal")
    ax.set_xlabel(r"$a_1$", labelpad=2)
    if ylabel:
        ax.set_ylabel(r"$a_2$", labelpad=1)
    for spine in ax.spines.values():
        spine.set_color("#667078")
    dots = ax.collections[-1]
    dots.set_facecolor("#0000FF")
    dots.set_alpha(ALPHA)
    dots.set_rasterized(True)
    np.testing.assert_array_equal(dots.get_offsets(), x)
    np.testing.assert_array_equal(dots.get_sizes(), [AREA])
    np.testing.assert_allclose(dots.get_facecolors(), [[0, 0, 1, ALPHA]], atol=1e-15)


def render(target: dict, samples: dict) -> None:
    common.style()
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
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
            fig.savefig(OUT / f"gmm40_seed2_ibolt256_sql256_{layout}.{ext}", dpi=360,
                        bbox_inches="tight", pad_inches=.045)
        plt.close(fig)


def aggregate(target: dict, gt: np.ndarray) -> tuple[list[dict], dict]:
    """Validate all 20 saved native-policy runs before averaging by method."""
    refcounts, _ = common.assignments(gt, target)
    threshold = np.maximum(10, .1 * refcounts)
    per_seed = []
    for method in METHODS:
        for seed in range(4):
            run = run_for(method, seed)
            cfg = read(run / "config.json")
            audit = read(run / "update_count_audit.json")
            assert cfg["seed"] == seed and cfg["steps"] == 100000 and cfg["batch"] == 256
            assert cfg["temperature"] == 1 and cfg["eval_samples"] == 10000
            assert audit["status"] == "passed" and audit["actor_updates"] == 100000
            assert audit["full_budget_completed"] and not audit["errors"]
            if method == "iBOLT":
                assert cfg["n"] == cfg["m"] == 256 and (cfg["width"], cfg["depth"]) == (256, 3)
                assert cfg["latent_mode"] == "random"
            elif method == "SQL":
                assert cfg["sql_kernel_particles"] == 256 and cfg["sql_kernel_update_ratio"] == .5
                assert cfg["sql_value_particles"] == 16 and (cfg["width"], cfg["depth"]) == (256, 2)
            other_target = read(run.parent / "target/definition.json")
            for key in ("means", "std", "weights", "scale", "bounded_mass"):
                np.testing.assert_array_equal(other_target[key], target[key])
            final = run / "evaluations/step_0100000"
            sample_path, metric_path = final / "samples.npy", final / "metrics.json"
            x, metric = np.load(sample_path, allow_pickle=False), read(metric_path)
            assert x.shape == (SAMPLES, 2) and np.isfinite(x).all() and (np.abs(x) <= 40.0001).all()
            counts, near = common.assignments(x, target)
            np.testing.assert_array_equal(counts, metric["mode_counts_3sigma"])
            np.testing.assert_allclose(threshold, metric["coverage_threshold"], atol=1e-12)
            assert int((counts >= threshold).sum()) == metric["mode_coverage"]
            assert abs(float(near.mean()) - metric["high_density_fraction"]) < 1e-12
            mmd2 = common.verify_mmd2(x, gt)
            assert abs(mmd2 - metric["mmd2"]) < 1e-10 and mmd2 > 0
            per_seed.append(dict(algorithm=method, seed=seed, mmd=math.sqrt(mmd2), mmd2=mmd2,
                                 mode_coverage=metric["mode_coverage"], near_fraction=float(near.mean()),
                                 run=str(run.relative_to(ROOT)), source_commit=cfg["source_git_commit"],
                                 sample_sha256=sha(sample_path), metric_sha256=sha(metric_path),
                                 config_sha256=sha(run / "config.json"),
                                 audit_sha256=sha(run / "update_count_audit.json")))
    summary = {}
    for method in METHODS:
        selected = [row for row in per_seed if row["algorithm"] == method]
        assert [row["seed"] for row in selected] == list(range(4))
        summary[method] = {
            key: {"mean": float(np.mean([row[key] for row in selected])),
                  "sample_sd": float(np.std([row[key] for row in selected], ddof=1))}
            for key in ("mmd", "mmd2", "mode_coverage", "near_fraction")
        }
    return per_seed, summary


def table(summary: dict) -> str:
    names = ["Ground truth", "SAC", "SQL", "MFPO", "DIPO", r"\textbf{iBOLT (ours)}"]
    def cell(method: str, key: str, digits: int) -> str:
        stats = summary[method][key]
        value = f"{stats['mean']:.{digits}f} $\\pm$ {stats['sample_sd']:.{digits}f}"
        return r"\textbf{" + value + "}" if method == "iBOLT" else value
    mmd = ["--"] + [cell(method, "mmd", 4) for method in METHODS]
    coverage = ["40.00"] + [cell(method, "mode_coverage", 2) for method in METHODS]
    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\sffamily",
        r"\caption{GMM40 sampling quality after 100{,}000 actor updates: mean $\pm$ sample "
        r"standard deviation over four training seeds (0--3). iBOLT uses $N=M=256$. Each run contributes 10{,}000 "
        r"native full-policy samples. MMD uses the first 2{,}048 samples and a shared, "
        r"independent ground-truth draw. Ground truth is a fixed reference, not a trained "
        r"method. Mode coverage counts reference components covered out of 40.}",
        r"\label{tab:gmm40_sampling_quality}",
        r"\footnotesize",
        r"\resizebox{0.85\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        "    Algorithm & " + " & ".join(names) + r" \\",
        r"    \midrule",
        "    MMD $\\downarrow$ & " + " & ".join(mmd) + r" \\",
        "    Mode coverage (/40) $\\uparrow$ & " + " & ".join(coverage) + r" \\",
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
    gt = samples["Ground truth"]
    per_seed, summary = aggregate(target, gt)
    render(target, samples)
    shutil.copyfile(OUT / "gmm40_seed2_ibolt256_sql256_main.pdf", OUT / "gmm40_comparison.pdf")
    (OUT / "gmm40_four_seed_mean_table.tex").write_text(table(summary))
    with (OUT / "per_seed.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader(); writer.writerows(per_seed)
    (OUT / "results.json").write_text(json.dumps({"figure_seed2": rows, "per_seed": per_seed,
                                                     "summary": summary, "provenance": provenance}, indent=2) + "\n")
    (OUT / "figure_caption.txt").write_text(
        "GMM40 final-policy samples at 100,000 actor updates (seed 2). "
        "Panels show ground truth, SAC, SQL, MFPO, DIPO, "
        "and iBOLT (N=M=256). "
        "Each panel contains 10,000 samples. iBOLT samples include random latent "
        "and learned conditional Gaussian noise; other methods use their native "
        "policy sampler. The separate table reports mean and sample standard "
        "deviation across seeds 0--3. Density contours show the same ground-truth "
        "distribution in every panel. Blue markers use alpha 0.2 and area 2.10 pt^2.\n")
    (OUT / "gmm40_figure.tex").write_text("\n".join([
        r"\begin{figure}[t]", r"\centering", r"\sffamily",
        r"\includegraphics[width=0.85\textwidth]{gmm40_comparison.pdf}",
        r"\caption{GMM40 final-policy samples after 100{,}000 actor updates (training seed 2). "
        r"iBOLT uses $N=M=256$. Each panel contains "
        r"10{,}000 samples; the separate table averages over training seeds 0--3.}",
        r"\label{fig:gmm40_sampling}", r"\end{figure}", ""]))
    print(json.dumps({"figure_seed2": rows, "summary": summary,
                      "figure": str(OUT / "gmm40_seed2_ibolt256_sql256_main.png")}, indent=2))


if __name__ == "__main__":
    main()
