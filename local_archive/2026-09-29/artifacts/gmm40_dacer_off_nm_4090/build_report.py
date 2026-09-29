"""Reproducible 4x4 DACER-off GMM40 report from verified saved samples."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT/"artifacts/gmm40_paper_ibolt_20260924"
sys.path.insert(0, str(PAPER))
import build_figure as base  # noqa: E402

CAMPAIGNS = {
    64: "gmm40-ibolt-dacer-off-nm64-100k-4seed-4090-20260925",
    128: "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925",
    256: "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925",
    512: "gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925",
}
OUT = HERE/"report"
NMS = (64, 128, 256, 512)
SEEDS = (0, 1, 2, 3)
ALPHA = .2
POINT_AREA_PT2 = 1.05
SOURCE_COMMITS = {
    64: "732169a535940f75d06ccd27b386c62c9f7835f6",
    128: "c429fbb2b22abd607e87b28d7577aa8c0ae4c536",
    256: "c429fbb2b22abd607e87b28d7577aa8c0ae4c536",
    512: "c429fbb2b22abd607e87b28d7577aa8c0ae4c536",
}
CONFIG_KEYS = ("method", "steps", "batch", "width", "depth", "temperature",
               "eval_samples", "mean_output_init_scale", "actor_learning_rate",
               "actor_log_std_bounds", "initial_log_std", "teacher_std_floor",
               "density_beta", "latent_mode", "loss", "Q", "training_data")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_inputs():
    files = {}
    arrays = {}; rows = []; histories = {}; configs = {}
    control = None; target = None; reference = None
    for nm in NMS:
        campaign = CAMPAIGNS[nm]
        root = HERE/"inputs"/campaign
        receipt = read(root/"LOCAL_COPY_VERIFIED.json")
        manifest = read(root/"manifest.json")
        status = read(root/"status.json")
        assert receipt["campaign"] == campaign
        assert status["phase"] == "completed" and not status["queued"] and not status["running"]
        assert manifest["source_commit"] == SOURCE_COMMITS[nm]
        assert manifest["plan"]["host"] == "vast1"
        assert manifest["plan"]["gpu_model"] == "RTX 4090"
        if nm == 64:
            assert manifest["plan"]["dacer_enabled"] is False
        files[str((root/"manifest.json").relative_to(HERE))] = digest(root/"manifest.json")
        definition = read(root/"results/target/definition.json")
        if target is None:
            target = definition
            reference = base.bounded_reference(target)
        else:
            for key in ("means", "std", "weights", "bounded_mass", "scale"):
                np.testing.assert_allclose(definition[key], target[key], rtol=0, atol=0)
        relevant = [j for j in manifest["jobs"] if j["nm"] == nm]
        assert len(relevant) == 4 and {j["seed"] for j in relevant} == set(SEEDS)
        for seed in SEEDS:
            job = next(j for j in relevant if j["seed"] == seed)
            name = f"ibolt_nm{nm}_s{seed}_100k"
            assert job["name"] == name
            run = root/"results"/name
            config = read(run/"config.json")
            state = read(run/"status.json")
            audit = read(run/"update_count_audit.json")
            latest = read(run/"latest.json")
            job_state = read(root/"jobs"/(name+".json"))
            wandb = read(run/"wandb_status.json")
            assert job_state["status"] == state["status"] == "completed"
            assert state["step"] == latest["step"] == 100000
            assert audit["status"] == "passed" and audit["actor_updates"] == 100000
            assert not audit["errors"] and audit["full_budget_completed"]
            assert config["n"] == config["m"] == nm and config["seed"] == seed
            assert config["source_git_commit"] == manifest["source_commit"]
            assert config["navigation"] is False
            assert wandb["url"].startswith("https://wandb.ai/OptiQ/gmm-trg/runs/")
            assert not wandb["warnings"], (name, wandb["warnings"])
            if control is None:
                control = config
            else:
                for key in CONFIG_KEYS:
                    assert config[key] == control[key], (key, name)
            assert config["method"] == "optiq_trg" and config["latent_mode"] == "random"
            assert config["batch"] == 256 and (config["width"], config["depth"]) == (256, 3)
            assert config["actor_log_std_bounds"] == [-5., -3.5]
            assert (config["initial_log_std"], config["teacher_std_floor"]) == (-4., .05)
            assert config["mean_output_init_scale"] == 16 and config["temperature"] == 1
            assert config["eval_samples"] == 10000
            assert read(run/"model_sizes.json")["total"] == 133636
            evaluation = run/"evaluations/step_0100000"
            for mode, suffix in (("full_policy", ""), ("mu_only", "_mu_only")):
                sample_path = evaluation/("samples"+suffix+".npy")
                metrics_path = evaluation/("metrics"+suffix+".json")
                sample = np.load(sample_path, allow_pickle=False)
                metric = read(metrics_path)
                assert sample.shape == (10000, 2) and np.isfinite(sample).all()
                assert (np.abs(sample) <= 40.0001).all()
                counts, near = base.assignments(sample, target)
                np.testing.assert_array_equal(counts, metric["mode_counts_3sigma"])
                assert abs(float(near.mean())-metric["high_density_fraction"]) < 1e-12
                assert metric["n_samples"] == 10000
                mmd2 = base.verify_mmd2(sample, reference)
                assert abs(mmd2-metric["mmd2"]) < 1e-10, (name, mode)
                row = dict(nm=nm, seed=seed, mode=mode, coverage=metric["mode_coverage"],
                           near=metric["high_density_fraction"], mmd2=metric["mmd2"],
                           mmd=math.sqrt(max(0, metric["mmd2"])),
                           wall_minutes=(job_state["finished"]-job_state["started"])/60,
                           campaign=campaign, source_commit=manifest["source_commit"],
                           wandb_url=wandb["url"])
                rows.append(row)
                arrays[(nm, seed, mode)] = sample
                for p in (sample_path, metrics_path):
                    files[str(p.relative_to(HERE))] = digest(p)
            histories[(nm, seed)] = [json.loads(line) for line in
                                    (run/"metrics.jsonl").read_text().splitlines() if line.strip()]
            assert histories[(nm, seed)][-1]["step"] == 100000
            configs[(nm, seed)] = config
    np.testing.assert_array_equal(reference,
                                  np.load(PAPER/"style_v5_full_policy/ground_truth_samples.npy"))
    return target, rows, arrays, histories, configs, files


def summarize(rows):
    output = []
    for mode in ("full_policy", "mu_only"):
        for nm in NMS:
            selected = [r for r in rows if r["mode"] == mode and r["nm"] == nm]
            assert len(selected) == 4
            result = dict(mode=mode, nm=nm, seeds=4)
            for key in ("mmd", "mmd2", "coverage", "near", "wall_minutes"):
                v = [x[key] for x in selected]
                result[key+"_mean"] = statistics.mean(v)
                result[key+"_sd"] = statistics.stdev(v)
            output.append(result)
    return output


def plot_panel(ax, samples, grid, title, *, xlabel=True, ylabel=True):
    base.panel(ax, samples, grid, title, xlabel=xlabel, ylabel=ylabel, size=POINT_AREA_PT2)
    points = ax.collections[-1]
    points.set_facecolor("#0000FF")
    points.set_alpha(ALPHA)
    points.set_rasterized(True)
    np.testing.assert_array_equal(points.get_offsets(), samples)
    np.testing.assert_array_equal(points.get_sizes(), np.array([POINT_AREA_PT2]))
    np.testing.assert_allclose(points.get_facecolors(), [[0, 0, 1, ALPHA]], atol=1e-15)


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/(stem+"."+ext), dpi=300, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


def plot_distributions(target, rows, arrays):
    grid = base.contour_grid(target)
    lookup = {(r["nm"], r["seed"], r["mode"]): r for r in rows}
    for mode in ("full_policy", "mu_only"):
        fig, axes = plt.subplots(4, 4, figsize=(13.8, 13.0))
        for i, nm in enumerate(NMS):
            for seed in SEEDS:
                row = lookup[(nm, seed, mode)]
                title = f"N=M={nm}, seed {seed}\n{row['coverage']}/40 · MMD {row['mmd']:.3f}"
                plot_panel(axes[i, seed], arrays[(nm, seed, mode)], grid, title,
                           xlabel=(i == 3), ylabel=(seed == 0))
        fig.suptitle("GMM40 · iBOLT · 100k updates · " + mode.replace("_", " "), y=.995)
        fig.subplots_adjust(left=.055, right=.99, bottom=.06, top=.945, hspace=.30, wspace=.20)
        save(fig, mode+"_all_seeds_alpha0p2_s1p05")
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.8))
        for i, nm in enumerate(NMS):
            row = lookup[(nm, 0, mode)]
            plot_panel(axes[i], arrays[(nm, 0, mode)], grid,
                       f"N=M={nm}\n{row['coverage']}/40 · MMD {row['mmd']:.3f}", ylabel=(i == 0))
        fig.subplots_adjust(left=.045, right=.995, bottom=.19, top=.85, wspace=.20)
        save(fig, mode+"_seed0_alpha0p2_s1p05")


def plot_learning(histories):
    colors = {64:"#4c78a8", 128:"#f58518", 256:"#54a24b", 512:"#b279a2"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for nm in NMS:
        sequences = [histories[(nm, seed)] for seed in SEEDS]
        steps = np.array([r["step"] for r in sequences[0]])
        assert all([r["step"] for r in seq] == list(steps) for seq in sequences)
        for ax, key in ((axes[0], "mmd2"), (axes[1], "mode_coverage")):
            vals = np.array([[math.sqrt(max(0, r[key])) if key == "mmd2" else r[key]
                              for r in seq] for seq in sequences], dtype=float)
            mean = vals.mean(axis=0); sd = vals.std(axis=0, ddof=1)
            ax.plot(steps, mean, color=colors[nm], label=f"N=M={nm}")
            ax.fill_between(steps, mean-sd, mean+sd, color=colors[nm], alpha=.14, lw=0)
    axes[0].set_ylabel("MMD ↓")
    axes[1].set_ylabel("Covered GT components / 40")
    axes[1].set_xlabel("Actor optimizer updates")
    axes[1].set_ylim(0, 40.8)
    axes[0].legend(ncol=2, frameon=False, fontsize=9)
    for ax in axes:ax.grid(alpha=.18, lw=.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT/("learning_curves."+ext), dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base.style()
    target, rows, arrays, histories, configs, files = check_inputs()
    summary = summarize(rows)
    plot_distributions(target, rows, arrays)
    plot_learning(histories)
    with (OUT/"per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (OUT/"results.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=2)+"\n")
    full = [r for r in summary if r["mode"] == "full_policy"]
    lines = ["# GMM40 DACER off: N=M 비교", "", "고정 Q GMM40에는 DACER 탐색 코드가 없습니다. 4090에서 학습한 16개 정책은 모두 DACER off이며, DACER on과의 비교 실험은 아닙니다.",
             "", "256×3 GELU, batch 256, T=1, mean-init=16, log σ=[−5,−3.5], 초기 −4, teacher floor .05. 시드0–3 각 100k actor 업데이트.",
             "", "Full-policy: 랜덤 latent와 학습된 conditional σ를 함께 샘플링. 각 패널 10,000개, 모든 점 파랑, alpha 0.2, 점 면적 1.05 pt².",
             "", "| N=M | Coverage /40 | MMD ↓ | 3σ 내 비율 | 실행시간/시드 |", "|---:|---:|---:|---:|---:|"]
    for r in full:
        lines.append(f"| {r['nm']} | {r['coverage_mean']:.2f} ± {r['coverage_sd']:.2f} | "
                     f"{r['mmd_mean']:.4f} ± {r['mmd_sd']:.4f} | "
                     f"{r['near_mean']:.1%} ± {r['near_sd']:.1%} | "
                     f"{r['wall_minutes_mean']:.1f} ± {r['wall_minutes_sd']:.1f}분 |")
    lines += ["", "MMD는 저장된 unbiased MMD²(첫 2,048표본, RBF 대역폭 1/2/5/10/20 평균)의 양수 제곱근을 시드별로 계산했습니다. Coverage는 GT 성분 중심의 3σ 범위와 최소 점유 수 기준입니다.",
              "", "동일한 100k 업데이트라도 N=M이 커지면 업데이트당 teacher Q 조회 수가 증가합니다. 64 대비 512의 조회 수는 8배입니다.",
              "", "그림: `full_policy_seed0_alpha0p2_s1p05.*`, `full_policy_all_seeds_alpha0p2_s1p05.*`, `learning_curves.*`. μ-only 분포는 이름에 `mu_only`를 붙여 별도로 보관합니다."]
    (OUT/"REPORT_KO.md").write_text("\n".join(lines)+"\n")
    provenance = dict(display_name="iBOLT", method="optiq_trg", dacer_enabled=False,
                      dacer_note="Not implemented in fixed-Q GMM40; both training sources use the same DACER-free learner.",
                      nms=NMS, seeds=SEEDS, actor_updates=100000, samples_per_panel=10000,
                      visualization=dict(primary="full_policy", alpha=ALPHA,
                                         point_area_pt2=POINT_AREA_PT2, color="#0000FF",
                                         contour_source="artifacts/gmm40_paper_ibolt_20260924/build_figure.py"),
                      training_sources=SOURCE_COMMITS, input_sha256=files,
                      reporter_sha256=digest(__file__), paper_plotter_sha256=digest(PAPER/"build_figure.py"),
                      reference_target_sha256=digest(PAPER/"style_v5_full_policy/target_definition.json"))
    (OUT/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"status":"completed", "full_policy_summary":full,
                      "figure":str(OUT/"full_policy_all_seeds_alpha0p2_s1p05.png")}, indent=2))


if __name__ == "__main__":
    main()
