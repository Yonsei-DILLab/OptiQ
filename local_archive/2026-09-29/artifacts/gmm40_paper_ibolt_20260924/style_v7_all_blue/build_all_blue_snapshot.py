"""All-blue version of the preserved full-policy GMM40 publication panels."""
from pathlib import Path

import numpy as np

import build_figure as base


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "style_v5_full_policy"
OUT = HERE / "style_v7_all_blue"


def panel(ax, samples, grid, title, **kwargs):
    base.panel(ax, samples, grid, title, **kwargs)
    points = ax.collections[-1]
    points.set_facecolor("#0000FF")
    np.testing.assert_array_equal(points.get_offsets(), samples)
    np.testing.assert_array_equal(points.get_facecolors(), [[0., 0., 1., 1.]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = base.read_json(SOURCE / "metrics_and_provenance.json")
    assert source["evaluation"]["ibolt"] == "full_policy_random_latent_conditional_sigma"
    target = base.read_json(SOURCE / "target_definition.json")
    reference = np.load(SOURCE / "ground_truth_samples.npy", allow_pickle=False)
    np.testing.assert_array_equal(reference, base.bounded_reference(target))
    data = {("Ground truth", 0): reference}
    hashes = {str((SOURCE / "ground_truth_samples.npy").relative_to(base.ROOT)):
              base.sha256(SOURCE / "ground_truth_samples.npy")}
    for row in source["per_seed"]:
        path = base.ROOT / row["run_path"] / "evaluations/step_0100000/samples.npy"
        relative = str(path.relative_to(base.ROOT))
        digest = base.sha256(path)
        assert digest == source["input_sha256"][relative], relative
        samples = np.load(path, allow_pickle=False)
        assert samples.shape == (10_000, 2) and np.isfinite(samples).all()
        data[(row["algorithm"], row["seed"])] = samples
        hashes[relative] = digest

    base.OUT = OUT
    base.style()
    grid = base.contour_grid(target)
    names = ["Ground truth"] + base.ORDER
    for wide in (False, True):
        fig, axes = base.plt.subplots(1 if wide else 2, 6 if wide else 3,
                                     figsize=(16.3, 3.2) if wide else (8.1, 6.0))
        for i, (ax, name) in enumerate(zip(axes.flat, names)):
            title = f"({chr(97 + i)}) {name}" + (" (ours)" if name == "iBOLT" else "")
            panel(ax, data[(name, 0)], grid, title)
        if wide:
            fig.subplots_adjust(left=.028, right=.995, bottom=.23, top=.87, wspace=.30)
        else:
            fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.94,
                                hspace=.32, wspace=.24)
        base.save(fig, f"gmm40_{'wide' if wide else 'main'}_seed0_all_blue")

    fig, axes = base.plt.subplots(4, 5, figsize=(13.1, 10.8))
    for seed in range(4):
        for i, name in enumerate(base.ORDER):
            ax = axes[seed, i]
            panel(ax, data[(name, seed)], grid, name if seed == 0 else "", size=.8)
            if i == 0:
                ax.set_ylabel(f"Seed {seed}\n" + r"$a_2$", fontsize=10)
    fig.subplots_adjust(left=.055, right=.995, bottom=.055, top=.96,
                        hspace=.22, wspace=.25)
    base.save(fig, "gmm40_all_four_seeds_all_blue", formats=("pdf", "png"), dpi=240)
    base.put_json(OUT / "provenance.json", {
        "source_revision": str(SOURCE),
        "source_provenance_sha256": base.sha256(SOURCE / "metrics_and_provenance.json"),
        "script_sha256": base.sha256(__file__),
        "base_plotting_script_sha256": base.sha256(HERE / "build_figure.py"),
        "target_definition_sha256": base.sha256(SOURCE / "target_definition.json"),
        "sample_sha256": hashes,
        "sampling": source["evaluation"],
        "primary_seed": 0,
        "actor_updates": 100000,
        "display": {
            "all_sample_colors": "#0000FF", "alpha": 1,
            "points_per_panel": 10000, "samples_and_order_unchanged": True,
            "contours_and_axes_unchanged": True, "color_classification_legend_removed": True,
            "plotted_offsets_and_uniform_blue_verified": True,
        },
    })
    (OUT / "README_KO.md").write_text(
        "# GMM40 전체 파란색 버전\n\n"
        "기존 100k full-policy 샘플 10,000개/패널을 순서와 위치 그대로 사용했습니다.\n"
        "iBOLT conditional sigma 및 SAC/DIPO/SQL/MFPO의 기존 확률적 생성 출력을 유지합니다.\n"
        "모든 점은 #0000FF, alpha=1입니다. 배경 GT 등고선과 축은 기존 스타일입니다.\n"
        "주 그림은 seed0, 별도 all_four_seeds 그림은 seed0..3입니다. 기존 지표와 원자료를 보존합니다.\n"
        "재생성: /tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_all_blue.py\n")
    print(f"Saved main, wide, and four-seed all-blue figures: {OUT}", flush=True)


if __name__ == "__main__":
    main()
