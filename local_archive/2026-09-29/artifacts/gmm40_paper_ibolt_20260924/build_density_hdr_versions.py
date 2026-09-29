"""Recolor preserved full-policy GMM40 figures by the target's density HDR.

No training or policy sampling. Thresholds use independent bounded-GT draws;
all plotted samples, order, axes, and background styling match revision v5.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

import build_figure as base


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "style_v5_full_policy"
OUT = HERE / "style_v6_density_hdr"
CALIBRATION_N = 1_000_000
VALIDATION_N = 1_000_000
CALIBRATION_SEED = 2026092501
VALIDATION_SEED = 2026092502
LEVELS = (0.99, 0.95)


def log_density(samples, target):
    """Analytic full mixture, with the archived box-conditioning constant."""
    samples = np.asarray(samples, dtype=np.float64)
    means = np.asarray(target["means"], dtype=np.float64)
    std = np.asarray(target["std"], dtype=np.float64)
    weights = np.asarray(target["weights"], dtype=np.float64)
    normalizers = np.log(weights) - 2 * np.log(std) - math.log(2 * math.pi)
    values = np.empty(len(samples), dtype=np.float64)
    for start in range(0, len(samples), 16384):
        x = samples[start:start + 16384]
        dist2 = ((x[:, None, 0] - means[None, :, 0]) ** 2
                 + (x[:, None, 1] - means[None, :, 1]) ** 2)
        terms = -dist2 / (2 * std[None, :] ** 2) + normalizers
        peak = terms.max(axis=1)
        lp = peak + np.log(np.exp(terms - peak[:, None]).sum(axis=1))
        lp -= math.log(target["bounded_mass"])
        lp[np.any(np.abs(x) > target["scale"], axis=1)] = -np.inf
        values[start:start + len(x)] = lp
    return values


def sample_bounded_target(target, n, seed):
    """Reject from the original mixture, not from separately truncated modes."""
    rng = np.random.default_rng(seed)
    means = np.asarray(target["means"], dtype=np.float64)
    std = np.asarray(target["std"], dtype=np.float64)
    weights = np.asarray(target["weights"], dtype=np.float64)
    out = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        count = min(65536, max(1024, n - filled))
        component = rng.choice(len(means), size=count, p=weights)
        draws = means[component] + std[component, None] * rng.normal(size=(count, 2))
        draws = draws[np.all(np.abs(draws) < target["scale"], axis=1)]
        take = min(len(draws), n - filled)
        out[filled:filled + take] = draws[:take]
        filled += take
    return out


def panel(ax, samples, lp, threshold, grid, title, size=1.05):
    xx, yy, relative, levels, _ = grid
    cmap = base.LinearSegmentedColormap.from_list(
        "viridis_muted_high", base.plt.colormaps["viridis"](np.linspace(.07, .80, 256)))
    ax.contour(xx, yy, relative, levels=levels, cmap=cmap,
               norm=base.PowerNorm(gamma=2.5, vmin=-140, vmax=0, clip=True),
               linewidths=.26, alpha=.68, zorder=1)
    colors = np.where(lp >= threshold, base.INSIDE_COLOR, base.OUTSIDE_COLOR)
    ax.scatter(samples[:, 0], samples[:, 1], s=size, c=colors, alpha=1,
               marker="o", linewidths=0, rasterized=False, zorder=2)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal")
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(pad=2)
    ax.set_title(title, pad=8)
    ax.set_xlabel(r"$a_1$", labelpad=2)
    ax.set_ylabel(r"$a_2$", labelpad=1)
    for spine in ax.spines.values():
        spine.set_color("#667078")


def legend(fig, probability):
    pct = round(100 * probability)
    handles = [base.Line2D([], [], color=color, marker="o", ls="", markersize=4,
                          label=f"{label} {pct}% GT highest-density region")
               for color, label in [(base.INSIDE_COLOR, "Inside"),
                                     (base.OUTSIDE_COLOR, "Outside")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=1.8,
               handletextpad=.55)


def make_figures(data, grid, probability, threshold):
    pct = round(probability * 100)
    base.OUT = OUT / f"hdr{pct}"
    base.OUT.mkdir(parents=True, exist_ok=True)
    names = ["Ground truth"] + base.ORDER
    for wide in (False, True):
        fig, axes = base.plt.subplots(1 if wide else 2, 6 if wide else 3,
                                     figsize=(16.3, 3.2) if wide else (8.1, 6.0))
        for i, (ax, name) in enumerate(zip(axes.flat, names)):
            samples, lp = data[(name, 0)]
            title = f"({chr(97 + i)}) {name}" + (" (ours)" if name == "iBOLT" else "")
            panel(ax, samples, lp, threshold, grid, title)
        if wide:
            fig.subplots_adjust(left=.028, right=.995, bottom=.23, top=.87, wspace=.30)
        else:
            fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.94,
                                hspace=.32, wspace=.24)
        legend(fig, probability)
        base.save(fig, f"gmm40_{'wide' if wide else 'main'}_seed0_hdr{pct}")
    fig, axes = base.plt.subplots(4, 5, figsize=(13.1, 10.8))
    for seed in range(4):
        for i, name in enumerate(base.ORDER):
            ax = axes[seed, i]
            samples, lp = data[(name, seed)]
            panel(ax, samples, lp, threshold, grid, name if seed == 0 else "", size=.8)
            if i == 0:
                ax.set_ylabel(f"Seed {seed}\n" + r"$a_2$", fontsize=10)
    fig.subplots_adjust(left=.055, right=.995, bottom=.055, top=.96,
                        hspace=.22, wspace=.25)
    legend(fig, probability)
    base.save(fig, f"gmm40_all_four_seeds_hdr{pct}", formats=("pdf", "png"), dpi=240)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = base.read_json(SOURCE / "metrics_and_provenance.json")
    assert old["evaluation"]["ibolt"] == "full_policy_random_latent_conditional_sigma"
    for name, digest in old["input_sha256"].items():
        assert base.sha256(base.ROOT / name) == digest, name
    target = base.read_json(SOURCE / "target_definition.json")
    reference = np.load(SOURCE / "ground_truth_samples.npy", allow_pickle=False)
    np.testing.assert_array_equal(reference, base.bounded_reference(target))
    data = {("Ground truth", 0): (reference, log_density(reference, target))}
    sample_hashes = {str((SOURCE / "ground_truth_samples.npy").relative_to(base.ROOT)):
                     base.sha256(SOURCE / "ground_truth_samples.npy")}
    for row in old["per_seed"]:
        file = base.ROOT / row["run_path"] / "evaluations/step_0100000/samples.npy"
        rel = str(file.relative_to(base.ROOT))
        assert base.sha256(file) == old["input_sha256"][rel], rel
        samples = np.load(file, allow_pickle=False)
        assert samples.shape == (10_000, 2) and np.isfinite(samples).all()
        assert np.all(np.abs(samples) <= 40.0001)
        data[(row["algorithm"], row["seed"])] = (samples, log_density(samples, target))
        sample_hashes[rel] = base.sha256(file)
    print("Verified all 20 original full-policy sample arrays and the unchanged GT panel.", flush=True)

    cal_lp = log_density(sample_bounded_target(target, CALIBRATION_N, CALIBRATION_SEED), target)
    val_lp = log_density(sample_bounded_target(target, VALIDATION_N, VALIDATION_SEED), target)
    thresholds = {}
    for probability in LEVELS:
        log_tau = float(np.quantile(cal_lp, 1 - probability))
        observed = float(np.mean(val_lp >= log_tau))
        assert abs(observed - probability) < .002, (probability, observed)
        thresholds[str(round(probability * 100))] = {
            "probability_mass": probability, "log_density_threshold": log_tau,
            "density_threshold": math.exp(log_tau),
            "calibration_inside_fraction": float(np.mean(cal_lp >= log_tau)),
            "independent_validation_inside_fraction": observed,
        }
    assert thresholds["99"]["log_density_threshold"] < thresholds["95"]["log_density_threshold"]
    rows = []
    for (name, seed), (samples, lp) in data.items():
        _, near = base.assignments(samples, target)
        inside99 = lp >= thresholds["99"]["log_density_threshold"]
        inside95 = lp >= thresholds["95"]["log_density_threshold"]
        assert np.all(~inside95 | inside99)
        rows.append({"algorithm": name, "seed": seed, "n": len(samples),
                     "outside_3sigma_percent": float(100 * np.mean(~near)),
                     "outside_hdr99_percent": float(100 * np.mean(~inside99)),
                     "outside_hdr95_percent": float(100 * np.mean(~inside95)),
                     "hdr99_changed_colors_vs_3sigma": int(np.sum(inside99 != near)),
                     "hdr95_changed_colors_vs_3sigma": int(np.sum(inside95 != near))})
    with (OUT / "sample_classification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"thresholds": thresholds, "seed0": [r for r in rows if r["seed"] == 0]}, indent=2), flush=True)

    base.style()
    grid = base.contour_grid(target)
    for probability in LEVELS:
        pct = str(round(100 * probability))
        make_figures(data, grid, probability, thresholds[pct]["log_density_threshold"])
        print(f"Saved {pct}% HDR: main, wide, and all-four-seed figures.", flush=True)
    manifest = {
        "source_revision": str(SOURCE), "source_provenance_sha256": base.sha256(SOURCE / "metrics_and_provenance.json"),
        "script_sha256": base.sha256(__file__), "base_plotting_script_sha256": base.sha256(HERE / "build_figure.py"),
        "target_definition_sha256": base.sha256(SOURCE / "target_definition.json"),
        "target": "Original GMM40 conditioned on the same [-40,40]^2 action box as v5",
        "density": "Sum of all 40 weighted GT Gaussian component densities / Z_box, via log-sum-exp",
        "calibration": {"n": CALIBRATION_N, "seed": CALIBRATION_SEED,
                        "threshold": "quantile of log p_GT(X) at 1-probability, X~p_GT",
                        "independent_of_plotted_samples": True},
        "validation": {"n": VALIDATION_N, "seed": VALIDATION_SEED},
        "thresholds": thresholds, "sampling": old["evaluation"],
        "sample_sha256": sample_hashes, "classification": rows,
        "display": {"blue": "#0000FF", "red": "#FF0000", "alpha": 1,
                    "all_points_retained": True, "same_sample_order_as_v5": True,
                    "same_density_contours_as_v5": True, "hdr95_subset_of_hdr99_verified": True},
        "metrics": "Archived MMD/3sigma mode coverage are unchanged. HDR is a new color rule only.",
    }
    base.put_json(OUT / "density_hdr_provenance.json", manifest)
    lines = ["# GMM40: 99% and 95% target highest-density regions", "",
             "기존 v5의 100k / seed0 / full-policy 샘플을 그대로 사용한 색상 비교입니다.",
             "Ground truth, SAC, SQL, MFPO, DIPO, iBOLT 순서를 유지했습니다.",
             "iBOLT conditional sigma, SAC sigma 및 각 baseline의 native stochastic 생성 결과를 유지했습니다.", "",
             "99%·95%는 GT 확률질량 기준입니다. 각 알고리즘의 빨간 표본 비율을 미리 고정하지 않습니다.",
             "GT에서 독립적으로 생성한 1,000,000개 log density의 하위 1%/5% 분위수로 임계값을 정했습니다.",
             "별도의 독립 GT 1,000,000개로 포함 확률을 검증했습니다. 같은 임계값을 모든 방법/시드에 적용합니다.",
             "밀도는 40개 성분의 가중합이며, 기존 평가와 동일한 action box 조건부 GT를 사용합니다.",
             "파랑: log p_GT(x) >= threshold. 빨강: log p_GT(x) < threshold.", "",
             "| HDR | Density threshold | 독립 GT 포함률 |", "|---|---:|---:|"]
    for pct, info in thresholds.items():
        lines.append(f"| {pct}% | {info['density_threshold']:.10g} | {100 * info['independent_validation_inside_fraction']:.4f}% |")
    lines += ["", "| seed0 | 기존 3sigma 밖 | 99% HDR 밖 | 95% HDR 밖 |",
              "|---|---:|---:|---:|"]
    for row in rows:
        if row["seed"] == 0:
            lines.append(f"| {row['algorithm']} | {row['outside_3sigma_percent']:.2f}% | {row['outside_hdr99_percent']:.2f}% | {row['outside_hdr95_percent']:.2f}% |")
    lines += ["", "MMD 및 기존 mode coverage는 다시 정의하거나 변경하지 않았습니다.",
              "GT에서도 HDR 밖 표본이 약 1%/5% 발생하므로 빨간 점을 확정적인 OOD로 해석하지 않습니다.",
              "이전 3sigma 그림·학습 결과·원자료는 보존했습니다.", "",
              "재생성: /tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_density_hdr_versions.py"]
    (OUT / "README_KO.md").write_text("\n".join(lines) + "\n")
    print(f"DONE {OUT}", flush=True)


if __name__ == "__main__":
    main()
