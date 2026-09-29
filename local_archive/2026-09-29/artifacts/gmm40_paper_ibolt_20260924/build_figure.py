"""Publication GMM40 panels from preserved, audited 100k checkpoints.

No training, checkpoint inference, sample filtering, or per-method seed selection.
Run with /tmp/optiq-antmaze-report-20260922/bin/python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / "artifacts/gmm40_5090_queue/results/results"
SEARCH = ROOT / "artifacts/gmm40_mu90_search"
ORDER = ["SAC", "SQL", "MFPO", "DIPO", "iBOLT"]
TABLE_ORDER = ["iBOLT", "SAC", "SQL", "MFPO", "DIPO"]
SAMPLES = 10_000
UPDATES = 100_000
REFERENCE_SEED = 20260917
BANDWIDTHS = [1., 2., 5., 10., 20.]
INSIDE_COLOR = "#0000FF"
OUTSIDE_COLOR = "#FF0000"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded_reference(target):
    """Exactly the archived Target.sample(n, 20260917, bounded=True)."""
    means, std = np.asarray(target["means"]), np.asarray(target["std"])
    rng = np.random.default_rng(REFERENCE_SEED)
    pieces, count = [], 0
    while count < SAMPLES:
        idx = rng.integers(40, size=max(1024, SAMPLES-count))
        x = means[idx] + std[idx, None] * rng.normal(size=(len(idx), 2))
        x = x[np.all(np.abs(x) < 40, axis=1)]
        pieces.append(x)
        count += len(x)
    return np.concatenate(pieces)[:SAMPLES].astype(np.float32)


def assignments(x, target):
    means, std = np.asarray(target["means"]), np.asarray(target["std"])
    # Identical nearest-standardized-center / 3-sigma definition as evaluation.py.
    distances = np.sum((x[:, None, :] - means[None, :, :])**2, axis=2) / std[None, :]**2
    labels = distances.argmin(axis=1)
    near = distances[np.arange(len(x)), labels] <= 9.0
    counts = np.bincount(labels[near], minlength=40)
    return counts, near


def squared_distance(x, y):
    return ((x[:, None, 0]-y[None, :, 0])**2 +
            (x[:, None, 1]-y[None, :, 1])**2)


def verify_mmd2(x, y):
    """Archived unbiased MMD^2, five RBF kernels, first 2,048 draws."""
    x, y = x[:2048].astype(np.float64), y[:2048].astype(np.float64)
    xx, yy, xy = squared_distance(x, x), squared_distance(y, y), squared_distance(x, y)
    values = []
    for h in BANDWIDTHS:
        kxx, kyy, kxy = np.exp(-xx/(2*h*h)), np.exp(-yy/(2*h*h)), np.exp(-xy/(2*h*h))
        values.append(float((kxx.sum()-len(x))/(len(x)*(len(x)-1)) +
                            (kyy.sum()-len(y))/(len(y)*(len(y)-1)) - 2*kxy.mean()))
    return float(np.mean(values))


def run_path(method, seed):
    if method != "iBOLT":
        return BASE / f"{method.lower()}_s{seed}_100k"
    if seed == 0:
        campaign = "gmm40-mu90-narrow64-256x3-capm3p5-20260921"
        return SEARCH / campaign / "results" / f"{campaign}-init16_nm64-s0"
    return SEARCH / "gmm40-mu90-replication-nm64-20260921/results" / f"mu90_rep_optiq_trg_s{seed}_100k"


def verify_inputs():
    target = read_json(BASE / "target/definition.json")
    reference = bounded_reference(target)
    refcounts, _ = assignments(reference, target)
    threshold = np.maximum(10, .1*refcounts)
    assert reference.shape == (SAMPLES, 2)
    assert np.isfinite(reference).all()
    manifest, per_seed, arrays, full_arrays, full_rows = {}, [], {}, {}, []
    ibolt_configs = []
    for method in ORDER:
        arrays[method] = {}
        for seed in range(4):
            run = run_path(method, seed)
            cfg = read_json(run / "config.json")
            audit = read_json(run / "update_count_audit.json")
            assert audit["status"] == "passed" and not audit["errors"]
            assert audit["actor_updates"] == UPDATES and audit["full_budget_completed"]
            assert cfg["steps"] == UPDATES and cfg["seed"] == seed
            assert cfg["batch"] == 256 and cfg["temperature"] == 1
            other_target = read_json(run.parent / "target/definition.json")
            for key in ("means", "std", "weights", "scale", "bounded_mass"):
                np.testing.assert_array_equal(other_target[key], target[key])
            suffix = "_mu_only" if method == "iBOLT" else ""
            evaluation = run / "evaluations/step_0100000"
            sample_path = evaluation / f"samples{suffix}.npy"
            metric_path = evaluation / f"metrics{suffix}.json"
            metrics = read_json(metric_path)
            x = np.load(sample_path, allow_pickle=False)
            assert x.shape == (SAMPLES, 2) and np.isfinite(x).all()
            assert (np.abs(x) <= 40.0001).all(), "Refuse to hide points outside the panel"
            counts, near = assignments(x, target)
            np.testing.assert_array_equal(counts, metrics["mode_counts_3sigma"])
            np.testing.assert_allclose(threshold, metrics["coverage_threshold"], atol=1e-12)
            assert int((counts >= threshold).sum()) == metrics["mode_coverage"]
            assert abs(near.mean()-metrics["high_density_fraction"]) < 1e-12
            recalculated = verify_mmd2(x, reference)
            assert abs(recalculated-metrics["mmd2"]) < 1e-10, (method, seed, recalculated, metrics["mmd2"])
            assert metrics["mmd2"] > 0
            arrays[method][seed] = x
            row = dict(algorithm=method, seed=seed, actor_updates=UPDATES,
                       evaluation_mode="mu_only_random_latent" if method == "iBOLT" else "native_generator_output",
                       mmd=math.sqrt(metrics["mmd2"]), mmd2=metrics["mmd2"],
                       covered_modes=metrics["mode_coverage"], total_modes=40,
                       near_fraction=metrics["high_density_fraction"],
                       source_commit=cfg["source_git_commit"], run_path=str(run.relative_to(ROOT)))
            per_seed.append(row)
            files = [sample_path, metric_path, run/"config.json", run/"update_count_audit.json", run.parent/"target/definition.json"]
            for p in files:
                manifest[str(p.relative_to(ROOT))] = sha256(p)
            if method == "iBOLT":
                assert cfg["width"] == 256 and cfg["depth"] == 3
                assert cfg["n"] == cfg["m"] == 64 and cfg["latent_mode"] == "random"
                assert cfg["actor_log_std_bounds"] == [-5., -3.5]
                assert cfg["initial_log_std"] == -4.0
                assert cfg["teacher_std_floor"] == .05 and cfg["mean_output_init_scale"] == 16
                ibolt_configs.append(cfg)
                full = np.load(evaluation/"samples.npy", allow_pickle=False)
                fm = read_json(evaluation/"metrics.json")
                assert full.shape == (SAMPLES, 2) and np.isfinite(full).all()
                assert abs(verify_mmd2(full, reference)-fm["mmd2"]) < 1e-10
                fc, _ = assignments(full, target)
                assert int((fc >= threshold).sum()) == fm["mode_coverage"]
                full_arrays[seed] = full
                full_rows.append(dict(seed=seed, mmd=math.sqrt(max(0, fm["mmd2"])),
                                      mmd2=fm["mmd2"], covered_modes=fm["mode_coverage"]))
                for p in (evaluation/"samples.npy", evaluation/"metrics.json"):
                    manifest[str(p.relative_to(ROOT))] = sha256(p)
            print(f"Verified {method} seed {seed}: MMD={row['mmd']:.6f}, coverage={row['covered_modes']}/40", flush=True)
    common_keys = ["n", "m", "batch", "width", "depth", "temperature", "actor_learning_rate", "actor_log_std_bounds", "initial_log_std", "latent_mode", "density_beta", "teacher_std_floor", "mean_output_init_scale"]
    for cfg in ibolt_configs[1:]:
        for key in common_keys:
            assert cfg[key] == ibolt_configs[0][key], key
    np.save(OUT/"ground_truth_samples.npy", reference)
    shutil.copyfile(BASE/"target/definition.json", OUT/"target_definition.json")
    return target, reference, arrays, per_seed, manifest, full_arrays, full_rows


def contour_grid(target):
    grid = np.linspace(-40, 40, 601)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx, yy], axis=-1)
    means, std = np.asarray(target["means"]), np.asarray(target["std"])
    # Stable analytic log density, bounded normalization is a constant shift.
    terms = -.5*np.sum(((points[..., None, :]-means)/std[:, None])**2, axis=-1)
    terms -= 2*np.log(std) + math.log(2*math.pi) + math.log(40)
    peak = terms.max(axis=-1)
    logp = peak + np.log(np.exp(terms-peak[..., None]).sum(axis=-1))
    logp -= math.log(target["bounded_mass"])
    relative = logp-logp.max()
    # Restore the original spatial contours exactly. Adjust only line styling
    # and the color transfer function, avoiding dense rings around each mode.
    levels = np.unique(np.r_[np.arange(-320., -18., 5.), [-18., -15., -12., -9., -7., -5., -3., -1.]])
    return xx, yy, relative, levels, target


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 12,
        "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, samples, grid, title, *, ylabel=True, xlabel=True, size=1.05):
    xx, yy, relative, levels, target = grid
    cmap = LinearSegmentedColormap.from_list(
        "viridis_muted_high", plt.colormaps["viridis"](np.linspace(.07, .80, 256)))
    ax.contour(xx, yy, relative, levels=levels, cmap=cmap,
               norm=PowerNorm(gamma=2.5, vmin=-140, vmax=0, clip=True),
               linewidths=.26, alpha=.68, zorder=1)
    # Every point is retained in original order; only its color is changed.
    _, near = assignments(samples, target)
    colors = np.where(near, INSIDE_COLOR, OUTSIDE_COLOR)
    ax.scatter(samples[:, 0], samples[:, 1], s=size, c=colors, alpha=1.0,
               marker="o", linewidths=0, rasterized=False, zorder=2)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal")
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(pad=2)
    ax.set_title(title, pad=8)
    if xlabel:
        ax.set_xlabel(r"$a_1$", labelpad=2)
    if ylabel:
        ax.set_ylabel(r"$a_2$", labelpad=1)
    for spine in ax.spines.values():
        spine.set_color("#667078")


def sample_legend(fig, fontsize=9):
    handles = [Line2D([], [], color=INSIDE_COLOR, marker="o", ls="", markersize=4,
                      label=r"Within any GT $3\sigma$ region"),
               Line2D([], [], color=OUTSIDE_COLOR, marker="o", ls="", markersize=4,
                      label=r"Outside all GT $3\sigma$ regions")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=2, frameon=False, fontsize=fontsize, columnspacing=2,
               handletextpad=.55)


def save(fig, stem, formats=("pdf", "svg", "png"), dpi=360):
    for ext in formats:
        fig.savefig(OUT/f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


def make_figures(target, reference, arrays, full_arrays):
    style()
    grid = contour_grid(target)
    names = ["Ground truth"] + ORDER
    data = [reference] + [arrays[name][0] for name in ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 6.0))
    for i, (ax, name, x) in enumerate(zip(axes.flat, names, data)):
        panel(ax, x, grid, f"({chr(97+i)}) {name}" + (" (ours)" if name == "iBOLT" else ""))
    fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.94, hspace=.32, wspace=.24)
    sample_legend(fig)
    save(fig, "gmm40_main_seed0")

    fig, axes = plt.subplots(1, 6, figsize=(16.3, 3.2))
    for i, (ax, name, x) in enumerate(zip(axes, names, data)):
        panel(ax, x, grid, f"({chr(97+i)}) {name}" + (" (ours)" if name == "iBOLT" else ""))
    fig.subplots_adjust(left=.028, right=.995, bottom=.23, top=.87, wspace=.30)
    sample_legend(fig)
    save(fig, "gmm40_wide_seed0")

    fig, axes = plt.subplots(4, 5, figsize=(13.1, 10.8))
    for seed in range(4):
        for i, name in enumerate(ORDER):
            ax = axes[seed, i]
            panel(ax, arrays[name][seed], grid, name if seed == 0 else "", size=.8)
            if i == 0:
                ax.set_ylabel(f"Seed {seed}\n"+r"$a_2$", fontsize=10)
    fig.subplots_adjust(left=.055, right=.995, bottom=.055, top=.96, hspace=.22, wspace=.25)
    sample_legend(fig)
    save(fig, "gmm40_all_four_seeds", formats=("pdf", "png"), dpi=240)

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3))
    for seed, ax in enumerate(axes):
        panel(ax, full_arrays[seed], grid, f"iBOLT full policy · seed {seed}")
    fig.subplots_adjust(left=.045, right=.99, bottom=.24, top=.86, wspace=.30)
    sample_legend(fig)
    save(fig, "gmm40_ibolt_full_policy_supplement", formats=("pdf", "png"))


def write_tables(rows, full_rows, ibolt_sampling="mu-only"):
    aggregate = {}
    for method in ORDER:
        group = [r for r in rows if r["algorithm"] == method]
        assert [r["seed"] for r in group] == list(range(4))
        aggregate[method] = {key: dict(mean=float(np.mean([r[key] for r in group])),
                                     sample_sd=float(np.std([r[key] for r in group], ddof=1)))
                             for key in ("mmd", "mmd2", "covered_modes", "near_fraction")}

    def formatted(method, key, seed=None):
        if seed is not None:
            row = next(r for r in rows if r["algorithm"] == method and r["seed"] == seed)
            return f"{row[key]:.4f}" if key == "mmd" else f"{row[key]}/40"
        val = aggregate[method][key]
        if key == "mmd":
            return f"{val['mean']:.4f} \\pm {val['sample_sd']:.4f}"
        return f"({val['mean']:.2f} \\pm {val['sample_sd']:.2f})/40"

    for seed, name in ((None, "gmm40_table.tex"), (0, "gmm40_table_seed0.tex")):
        caption = ("Sample quality on the bounded GMM40 target after 100,000 actor updates. "
                   + ("Values are mean $\\pm$ sample standard deviation over four training seeds (0--3). " if seed is None else "Values correspond to the seed-0 policies shown in Figure~\\ref{fig:gmm40}. ")
                   + ("iBOLT uses its full conditional Gaussian policy, including learned $\\sigma$ and fresh Gaussian latents; baselines use their native stochastic generators."
                      if ibolt_sampling == "full-policy" else
                      "iBOLT uses $\\mu$-only outputs with fresh Gaussian latents; baselines use their native generator outputs."))
        lines = [r"% Requires \usepackage{booktabs,graphicx}", r"\begin{table}[h]", r"\centering",
                 r"\caption{"+caption+"}", r"\label{tab:gmm40"+("-seed0" if seed == 0 else "")+"}",
                 r"\footnotesize", r"\resizebox{0.95\textwidth}{!}{%", r"\begin{tabular}{lccccc}",
                 r"\toprule", r"Algorithm & \textbf{iBOLT (ours)} & SAC & SQL & MFPO & DIPO \\", r"\midrule"]
        for key, label in (("mmd", r"MMD $\downarrow$"), ("covered_modes", r"Mode coverage $\uparrow$")):
            vals = []
            for method in TABLE_ORDER:
                value = formatted(method, key, seed)
                vals.append("$"+(r"\mathbf{"+value+"}" if method == "iBOLT" else value)+"$")
            lines.append(label+" & "+" & ".join(vals)+r" \\")
        lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\vspace{0.2em}",
                  r"\parbox{0.95\textwidth}{\scriptsize",
                  r"MMD is $\sqrt{\max(\widehat{\mathrm{MMD}}_u^2,0)}$, computed per seed using the first 2,048 of 10,000 draws and an equal mixture of RBF kernels with bandwidths $1,2,5,10,20$. ",
                  r"Coverage counts components among 40 using all 10,000 draws: nearest-center distance at most $3\sigma$, with at least $\max(10,0.1n_j^{\mathrm{GT}})$ assigned samples per component. ",
                  r"iBOLT uses the GMM40-tuned $256\times3$ setting; equal actor-update budgets do not imply equal computation.",
                  "}", r"\end{table}"]
        (OUT/name).write_text("\n".join(lines)+"\n")
    full_aggregate = {key: dict(mean=float(np.mean([r[key] for r in full_rows])),
                                sample_sd=float(np.std([r[key] for r in full_rows], ddof=1)))
                      for key in ("mmd", "mmd2", "covered_modes")}
    return aggregate, full_aggregate


def documents(aggregate, rows, full_rows, full_aggregate, manifest, ibolt_sampling="mu-only"):
    ibolt_caption = (r"iBOLT samples fresh $z\sim\mathcal{N}(0,I)$ and then samples from its learned box-truncated conditional Gaussian, including $\sigma(s,z)$; "
                     if ibolt_sampling == "full-policy" else
                     r"iBOLT displays $\mu(s,z)$ for fresh $z\sim\mathcal{N}(0,I)$, with conditional Gaussian noise removed; ")
    caption = (r"Ground-truth samples and outputs of SAC, SQL (SVGD), MFPO, DIPO, and iBOLT on the bounded GMM40 target. "
               r"All learned models are shown after 100,000 actor updates using training seed 0. "
               r"Each panel displays all 10,000 evaluation draws without filtering; identical contours show the ground-truth log density. "
               r"Blue samples lie within $3\sigma$ of at least one ground-truth component center; red samples lie outside all such regions. "
               + ibolt_caption +
               r"the baselines retain their native generator outputs. All axes use the physical action coordinates. "
               r"Aggregate results over four training seeds are reported separately in Table~\ref{tab:gmm40}.")
    (OUT/"gmm40_figure.tex").write_text("\n".join([
        r"\begin{figure*}[t]", r"\centering", r"\includegraphics[width=0.95\textwidth]{gmm40_main_seed0.pdf}",
        r"\caption{"+caption+"}", r"\label{fig:gmm40}", r"\end{figure*}", ""]))
    (OUT/"figure_caption.txt").write_text(caption+"\n")
    provenance = dict(
        algorithm_display_name="iBOLT", historical_algorithm_id="optiq_trg",
        selection="User selected tuned 100k 256x3 setting; one fixed seed (0) for all primary panels; all seeds 0..3 in table/supplement.",
        ibolt=dict(width=256, depth=3, n=64, m=64, mean_head_variance_scale=16,
                   log_sigma_bounds=[-5., -3.5], initial_log_sigma=-4., teacher_std_floor=.05,
                   batch=256, temperature=1., density_beta=1., actor_learning_rate=.0003, latent="fresh normal",
                   source_seed0="e0f1ba03e1b7c67eae76b098dcd3c8c34ba14da1", source_seeds1_to3="29e6fc98b278a41b4f3b365feddd0e931c808f4c"),
        baseline_source="87d5d8ff210569ace7e59bd8a53ad02141b67f0a", actor_updates=UPDATES,
        evaluation=dict(reference_seed=REFERENCE_SEED, displayed_samples_per_panel=SAMPLES,
                        coverage_samples=SAMPLES, mmd_samples=2048, kernel_bandwidths=BANDWIDTHS,
                        estimator="sqrt(max(unbiased multi-bandwidth MMD^2,0)), calculated per seed before aggregation",
                        statistic="sample SD, ddof=1; training seeds 0,1,2,3",
                        ibolt="full_policy_random_latent_conditional_sigma" if ibolt_sampling == "full-policy" else "mu_only_random_latent",
                        baselines="native_generator_output"),
        verified=dict(all_20_optimizer_audits=True, all_20_raw_sample_metrics_recomputed=True,
                      ibolt_full_policy_supplement_recomputed=True, target_values_equal=True,
                      ibolt_hyperparameters_equal_across_seeds=True),
        per_seed=rows, aggregate=aggregate, full_policy_supplement=dict(per_seed=full_rows, aggregate=full_aggregate),
        input_sha256=manifest, plotting_script_sha256=sha256(__file__),
        visualization=dict(revision="v5_full_policy" if ibolt_sampling == "full-policy" else "v4_pure_blue_red", inside_3sigma_color=INSIDE_COLOR,
                           outside_3sigma_color=OUTSIDE_COLOR, threshold_squared_mahalanobis=9,
                           sample_order="unchanged", sample_alpha=1.0, all_samples_retained=True,
                           contours="original target log-density contours; subtract common peak",
                           colormap="viridis [0.07, 0.80]", contour_normalization="PowerNorm(gamma=2.5, vmin=-140, vmax=0)",
                           contour_linewidth=.26, contour_alpha=.68))
    put_json(OUT/"metrics_and_provenance.json", provenance)
    output_description = ("그림과 표 모두 iBOLT는 fresh Gaussian z와 학습된 conditional σ를 포함한 box-truncated Gaussian full-policy 샘플입니다. SAC는 tanh Gaussian에서 μ+σε를 샘플링하고, DIPO는 초기 Gaussian 및 100-step reverse diffusion 잡음(noise_ratio=1, 마지막 t=0 잡음 제외)을 포함합니다. SQL은 Gaussian latent를 actor에 통과시키며 MFPO는 Gaussian 초기값에서 두 flow step을 사용합니다. SQL/MFPO에 별도 출력 Gaussian 잡음을 더하지 않습니다."
                          if ibolt_sampling == "full-policy" else
                          "그림과 표 모두 iBOLT는 fresh Gaussian z의 μ-only, baseline은 native generator output입니다. μ-only 결과를 full-policy density 성능으로 해석하지 마세요.")
    lines = ["# GMM40 논문 피겨 · iBOLT", "",
             "Ground truth → SAC → SQL → MFPO → DIPO → iBOLT 순서입니다. 본문은 2×3이며 1×6 대안도 저장했습니다.", "",
             "- 본문: `gmm40_main_seed0.pdf` / `.svg` / `.png` (seed 0, 모든 표본 10,000개)",
             "- 본문 표: `gmm40_table.tex` (학습 seed 0~3 평균 ± 표본 표준편차)",
             "- seed 0 그림과 일대일 대응하는 표: `gmm40_table_seed0.tex`", 
             "- 캡션과 삽입 코드: `gmm40_figure.tex`", 
             "- 모든 시드 보조 그림: `gmm40_all_four_seeds.pdf` / `.png`", 
             "- iBOLT σ 포함 보조 그림: `gmm40_ibolt_full_policy_supplement.pdf` / `.png`", "",
             "전부 100k actor updates. iBOLT는 사용자 선택의 GMM40 튜닝 설정(256×3, N=M64, mean-head variance scale16, logσ[-5,-3.5], 초기-4, teacher floor.05)입니다. 기본 256×2 / logσ[-5,-1] 결과 또는 500k 결과와 섞지 않았습니다. baseline은 native architecture/optimizer를 유지하며 계산량·Q 질의량이 같다는 비교는 아닙니다.", "",
             output_description + " GT 성분 중 하나의 3σ 안이면 순수 파랑, 모두의 3σ 밖이면 순수 빨강입니다. 모든 표본을 원래 순서로 표시하며 근접 여부로 걸러내지 않았습니다. target log-density 등고선과 표본 색상·크기는 유지했습니다. 시드를 섞거나 모델별로 좋은 시드를 고르지 않았습니다.", "",
             "MMD는 기존 unbiased MMD² 추정값을 시드별 제곱근 변환한 값입니다. 원시 표본에서 재계산해 원래 저장값과 1e-10 이내 일치를 확인했습니다. 5개 RBF bandwidth(1,2,5,10,20) 평균, 원래 순서 첫 2048개 표본과 공통 reference를 사용합니다. MMD²도 JSON에 보존합니다. Ground truth는 기존 target sampler의 bounded=True, seed20260917로 재생성했습니다.", "",
             "Coverage는 10,000개 전체에서 최근접 성분의 3σ 이내 표본 수가 max(10, reference 성분 점유 수×.1) 이상인 성분 수입니다. 분포의 엄밀한 국소 극대점 개수는 아닙니다. target means/std/weights/scale/bounded_mass와 20개 optimizer audit를 확인했습니다.", "",
             "| Algorithm | MMD ↓ | Coverage /40 ↑ | MMD² (archived) |", "|---|---:|---:|---:|"]
    for method in TABLE_ORDER:
        a=aggregate[method]
        lines.append(f"| {method} | {a['mmd']['mean']:.4f} ± {a['mmd']['sample_sd']:.4f} | {a['covered_modes']['mean']:.2f} ± {a['covered_modes']['sample_sd']:.2f} | {a['mmd2']['mean']:.6f} ± {a['mmd2']['sample_sd']:.6f} |")
    lines += ["", "iBOLT 각 seed coverage: 40, 35, 40, 40. 본문 seed0의 40/40을 4시드 전부의 결과로 일반화하지 않습니다.", "",
              "보존된 OptiQ run 이름/소스/설정은 provenance를 위해 유지하고 논문 및 새 보고의 명칭만 iBOLT로 사용합니다.", "",
              "재생성: `/tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_figure.py --restyle-only --ibolt-sampling " + ibolt_sampling + " --output-dir " + str(OUT.relative_to(ROOT)) + "`"]
    (OUT/"README_KO.md").write_text("\n".join(lines)+"\n")


def load_verified_inputs(source_dir):
    """A style-only pass carries forward already verified numerical metrics."""
    old = read_json(source_dir/"metrics_and_provenance.json")
    for name, digest in old["input_sha256"].items():
        assert sha256(ROOT/name) == digest, name
    target = read_json(source_dir/"target_definition.json")
    reference = np.load(source_dir/"ground_truth_samples.npy", allow_pickle=False)
    arrays, full_arrays = {}, {}
    for method in ORDER:
        arrays[method] = {}
        for seed in range(4):
            evaluation = run_path(method, seed)/"evaluations/step_0100000"
            suffix = "_mu_only" if method == "iBOLT" else ""
            x = np.load(evaluation/f"samples{suffix}.npy", allow_pickle=False)
            assert x.shape == (SAMPLES, 2)
            _, near = assignments(x, target)
            row = next(r for r in old["per_seed"] if r["algorithm"] == method and r["seed"] == seed)
            assert abs(float(near.mean())-row["near_fraction"]) < 1e-12
            arrays[method][seed] = x
            if method == "iBOLT":
                full_arrays[seed] = np.load(evaluation/"samples.npy", allow_pickle=False)
    if OUT != source_dir:
        shutil.copyfile(source_dir/"ground_truth_samples.npy", OUT/"ground_truth_samples.npy")
        shutil.copyfile(source_dir/"target_definition.json", OUT/"target_definition.json")
    return target, reference, arrays, old["per_seed"], old["input_sha256"], full_arrays, old["full_policy_supplement"]["per_seed"]


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--restyle-only", action="store_true")
    parser.add_argument("--ibolt-sampling", choices=["mu-only", "full-policy"], default="mu-only")
    args = parser.parse_args()
    source_dir = OUT
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_verified_inputs(source_dir) if args.restyle_only else verify_inputs()
    target, reference, arrays, rows, manifest, full_arrays, full_rows = inputs
    mu_arrays = arrays["iBOLT"].copy()
    if args.ibolt_sampling == "full-policy":
        refcounts, _ = assignments(reference, target)
        threshold = np.maximum(10, .1*refcounts)
        for row in rows:
            if row["algorithm"] != "iBOLT":
                continue
            seed = row["seed"]
            x = full_arrays[seed]
            fm = read_json(run_path("iBOLT", seed)/"evaluations/step_0100000/metrics.json")
            assert x.shape == (SAMPLES, 2) and np.isfinite(x).all()
            assert (np.abs(x) <= 40.0001).all()
            assert not np.array_equal(x, mu_arrays[seed])
            counts, near = assignments(x, target)
            np.testing.assert_array_equal(counts, fm["mode_counts_3sigma"])
            assert int((counts >= threshold).sum()) == fm["mode_coverage"]
            assert abs(float(near.mean())-fm["high_density_fraction"]) < 1e-12
            assert abs(verify_mmd2(x, reference)-fm["mmd2"]) < 1e-10
            row.update(evaluation_mode="full_policy_random_latent_conditional_sigma",
                       mmd=math.sqrt(max(0, fm["mmd2"])), mmd2=fm["mmd2"],
                       covered_modes=fm["mode_coverage"], near_fraction=fm["high_density_fraction"])
            arrays["iBOLT"][seed] = x
    aggregate, full_aggregate = write_tables(rows, full_rows, args.ibolt_sampling)
    make_figures(target, reference, arrays, full_arrays)
    if args.ibolt_sampling == "full-policy":
        grid = contour_grid(target)
        fig, axes = plt.subplots(1, 3, figsize=(8.1, 3.35))
        for ax, x, title in zip(axes, [reference, mu_arrays[0], full_arrays[0]],
                                ["Ground truth", r"iBOLT: $\mu$ only", r"iBOLT: full policy ($\mu,\sigma$)"]):
            panel(ax, x, grid, title)
        fig.subplots_adjust(left=.055, right=.99, bottom=.24, top=.87, wspace=.24)
        sample_legend(fig)
        save(fig, "gmm40_ibolt_mu_vs_full_seed0")
    documents(aggregate, rows, full_rows, full_aggregate, manifest, args.ibolt_sampling)
    if args.restyle_only:
        current = read_json(OUT/"metrics_and_provenance.json")
        old = read_json(source_dir/"metrics_and_provenance.json")
        if args.ibolt_sampling == "mu-only":
            assert current["per_seed"] == old["per_seed"]
            assert current["aggregate"] == old["aggregate"]
            for name in ("gmm40_table.tex", "gmm40_table_seed0.tex"):
                assert (OUT/name).read_bytes() == (source_dir/name).read_bytes()
        else:
            for method in ORDER:
                if method != "iBOLT":
                    assert current["aggregate"][method] == old["aggregate"][method]
            for key in ("mmd", "mmd2", "covered_modes"):
                assert current["aggregate"]["iBOLT"][key] == old["full_policy_supplement"]["aggregate"][key]
        current["style_validation"] = dict(previous_verification=str(source_dir/"metrics_and_provenance.json"),
                                           raw_input_hashes_rechecked=True, near_masks_rechecked=True,
                                           metrics_and_latex_tables_unchanged=args.ibolt_sampling == "mu-only",
                                           ibolt_full_policy_metrics_recomputed=args.ibolt_sampling == "full-policy",
                                           baseline_samples_and_metrics_unchanged=True)
        put_json(OUT/"metrics_and_provenance.json", current)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
