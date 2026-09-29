"""Uniform-alpha variants of the same preserved full-policy GMM40 samples."""
from pathlib import Path

import numpy as np

import build_all_blue as blue


base = blue.base
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "style_v5_full_policy"
OUT = HERE / "style_v9_all_blue_alpha"
ALPHAS = (0.1, 0.2, 0.4)


def panel(ax, samples, grid, title, alpha, **kwargs):
    blue.panel(ax, samples, grid, title, **kwargs)
    points = ax.collections[-1]
    points.set_alpha(alpha)
    np.testing.assert_allclose(points.get_facecolors(), [[0., 0., 1., alpha]],
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(points.get_offsets(), samples)


def render(alpha, data, grid):
    tag = str(alpha).replace(".", "p")
    base.OUT = OUT / f"alpha_{tag}"
    base.OUT.mkdir(parents=True, exist_ok=True)
    names = ["Ground truth"] + base.ORDER
    for wide in (False, True):
        fig, axes = base.plt.subplots(1 if wide else 2, 6 if wide else 3,
                                     figsize=(16.3, 3.2) if wide else (8.1, 6.0))
        for i, (ax, name) in enumerate(zip(axes.flat, names)):
            title = f"({chr(97 + i)}) {name}" + (" (ours)" if name == "iBOLT" else "")
            panel(ax, data[(name, 0)], grid, title, alpha)
        if wide:
            fig.subplots_adjust(left=.028, right=.995, bottom=.23, top=.87, wspace=.30)
        else:
            fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.94,
                                hspace=.32, wspace=.24)
        base.save(fig, f"gmm40_{'wide' if wide else 'main'}_seed0_alpha{tag}")
    fig, axes = base.plt.subplots(4, 5, figsize=(13.1, 10.8))
    for seed in range(4):
        for i, name in enumerate(base.ORDER):
            ax = axes[seed, i]
            panel(ax, data[(name, seed)], grid, name if seed == 0 else "", alpha, size=.8)
            if i == 0:
                ax.set_ylabel(f"Seed {seed}\n" + r"$a_2$", fontsize=10)
    fig.subplots_adjust(left=.055, right=.995, bottom=.055, top=.96,
                        hspace=.22, wspace=.25)
    base.save(fig, f"gmm40_all_four_seeds_alpha{tag}", formats=("pdf", "png"), dpi=240)
    print(f"Saved alpha={alpha:g}: main, wide, and all-four-seed figures.", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", choices=ALPHAS, default=list(ALPHAS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = base.read_json(SOURCE / "metrics_and_provenance.json")
    assert source["evaluation"]["ibolt"] == "full_policy_random_latent_conditional_sigma"
    assert blue.POINT_AREA_SCALE == 2 / 3
    target = base.read_json(SOURCE / "target_definition.json")
    reference_path = SOURCE / "ground_truth_samples.npy"
    reference = np.load(reference_path, allow_pickle=False)
    np.testing.assert_array_equal(reference, base.bounded_reference(target))
    data = {("Ground truth", 0): reference}
    hashes = {str(reference_path.relative_to(base.ROOT)): base.sha256(reference_path)}
    for row in source["per_seed"]:
        path = base.ROOT / row["run_path"] / "evaluations/step_0100000/samples.npy"
        relative = str(path.relative_to(base.ROOT))
        digest = base.sha256(path)
        assert digest == source["input_sha256"][relative], relative
        samples = np.load(path, allow_pickle=False)
        assert samples.shape == (10_000, 2) and np.isfinite(samples).all()
        data[(row["algorithm"], row["seed"])] = samples
        hashes[relative] = digest
    base.style()
    grid = base.contour_grid(target)
    for alpha in args.alphas:
        render(alpha, data, grid)
    saved_alphas = [alpha for alpha in ALPHAS
                    if (OUT / f"alpha_{str(alpha).replace('.', 'p')}" /
                        f"gmm40_main_seed0_alpha{str(alpha).replace('.', 'p')}.png").is_file()]
    base.put_json(OUT / "provenance.json", {
        "source_revision": str(SOURCE),
        "source_provenance_sha256": base.sha256(SOURCE / "metrics_and_provenance.json"),
        "script_sha256": base.sha256(__file__),
        "dependency_sha256": {name: base.sha256(HERE / name)
                              for name in ("build_all_blue.py", "build_figure.py")},
        "target_definition_sha256": base.sha256(SOURCE / "target_definition.json"),
        "sample_sha256": hashes, "sampling": source["evaluation"],
        "primary_seed": 0, "actor_updates": 100000,
        "display": {
            "all_sample_colors": "#0000FF", "uniform_alpha_variants": saved_alphas,
            "main_point_area_pt2": 1.05 * blue.POINT_AREA_SCALE,
            "supplement_point_area_pt2": .8 * blue.POINT_AREA_SCALE,
            "points_per_panel": 10000, "all_samples_retained_in_original_order": True,
            "same_alpha_for_all_methods_and_ground_truth": True,
            "contours_axes_and_sizes_unchanged_from_v8": True,
            "actual_plotted_offsets_and_rgba_verified": True,
        },
    })
    (OUT / "README_KO.md").write_text(
        "# GMM40 전체 파란색 alpha 비교\n\n"
        "동일한 100k full-policy 샘플, 동일한 순서, 10,000개/패널을 사용합니다.\n"
        "주 그림은 seed0, 별도 supplement는 seed0..3입니다.\n"
        "모든 패널에 같은 alpha를 적용하며 0.1, 0.2, 0.4 버전을 제공합니다.\n"
        "점 색 #0000FF, 주 그림 점 면적 s=0.70 pt², 기존 등고선과 축을 유지합니다.\n"
        "임계값에 따른 표본 제거 없이 모든 샘플을 그립니다. 기존 지표는 유지됩니다.\n"
        "PDF/SVG는 점별 투명도를 유지하는 벡터 파일입니다.\n"
        "재생성: /tmp/optiq-antmaze-report-20260922/bin/python artifacts/gmm40_paper_ibolt_20260924/build_alpha_versions.py\n")


if __name__ == "__main__":
    main()
