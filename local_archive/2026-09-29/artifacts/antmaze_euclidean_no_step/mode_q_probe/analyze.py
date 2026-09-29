"""Read-only analysis of post-hoc mode/Q probes. Writes only separate reports.

Run with the compatible report runtime:
  /tmp/optiq-antmaze-report-20260922/bin/python <this-file>
No learner, simulator, checkpoint restoration, or training imports are used.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = (250112, 500224, 750080)
COLORS = {"left": "#326dc4", "lower": "#326dc4", "right": "#e58b2a",
          "upper": "#e58b2a", "uncommitted": "#969da7", "both": "#8754ac"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all()
    return {"n": int(x.size), "mean": float(x.mean()) if x.size else None,
            "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "min": float(x.min()) if x.size else None,
            "max": float(x.max()) if x.size else None}


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def mean_sd(s):
    return f"{fmt(s['mean'])} ± {fmt(s['sd'])}"


def ordered_routes(task, found):
    preferred = ("left", "right") if task == "v3" else ("lower", "upper")
    return [x for x in (*preferred, "both", "uncommitted") if x in found]


def process_task(folder):
    summary = json.loads((folder / "summary.json").read_text())
    bank_path = folder / "reference_state_action_bank.npz"
    with np.load(bank_path, allow_pickle=False) as raw:
        b = {k: raw[k] for k in raw.files}
    task = summary["task"]
    offsets = b["offsets"].astype(int)
    lengths = np.diff(offsets)
    routes = b["routes"].astype(str)
    assert offsets[0] == 0 and offsets[-1] == len(b["observations"])
    assert len(routes) == len(lengths) == len(b["goals"])
    assert (lengths > 0).all()
    for key in ("observations", "actions", "rewards", "return_to_go"):
        assert np.isfinite(b[key]).all(), key
    initial = b["observations"][offsets[:-1]]
    assert np.array_equal(initial, np.repeat(initial[:1], len(initial), axis=0))
    rtg_errors = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        rtg = 0.
        for j in range(hi - 1, lo - 1, -1):
            rtg = float(b["rewards"][j]) + .99 * rtg
            rtg_errors.append(abs(rtg - float(b["return_to_go"][j])))
    assert max(rtg_errors) < 1e-6
    order = ordered_routes(task, set(routes))
    out = {"task": task, "training_source": summary["training_source"],
           "checkpoint_sha256": summary["checkpoint_sha256"],
           "probe_script_sha256": summary["script_sha256"],
           "probe_verification": summary.get("verification"),
           "reference_reproduction_max_xy_error": summary.get("reference_reproduction_max_xy_error"),
           "scope": "250k policy reference state/action bank; checkpoint critics evaluate identical bank",
           "validation": {"identical_initial_observation": True,
                          "rtg_max_reconstruction_error": max(rtg_errors),
                          "initial_state_equality_scope": "all 29 actor observations equal; probe fixes full simulator state"},
           "routes": order, "same_initial": {}, "cross_checkpoint": {},
           "first100_q_curves": {}, "input_sha256": {
               "summary.json": sha(folder / "summary.json"), bank_path.name: sha(bank_path)}}
    initial_ix = offsets[:-1]
    initial_q = b[f"q_{STEPS[0]}"][:, initial_ix].mean(0)
    initial_g = b["return_to_go"][initial_ix]
    out["same_initial_all_episodes"] = {
        "q_mean": stats(initial_q), "actual_rtg": stats(initial_g),
        "actual_rtg_minus_q": stats(initial_g - initial_q),
        "note": "Matched-time rollout comparison. Not by itself proof of directional Q bias or collapse cause."}
    if summary.get("verification"):
        assert all(summary["verification"][k] for k in (
            "checkpoints_unchanged", "restored_states_unchanged", "no_training_updates"))
    for side in order:
        ids = np.flatnonzero(routes == side)
        ix = offsets[ids]
        assert len(ids) == summary["reference_groups"][side]["n"]
        q = b[f"q_{STEPS[0]}"]
        out["same_initial"][side] = {
            "n": len(ids), "q_mean": stats(q[:, ix].mean(0)),
            "q1": stats(q[0, ix]), "q2": stats(q[1, ix]),
            "actual_rtg": stats(b["return_to_go"][ix]),
            "actual_undiscounted": stats([b["rewards"][offsets[i]:offsets[i + 1]].sum() for i in ids]),
            "successes": int((b["goals"][ids] > 0).sum())}
        curve = []
        for t in range(101):
            selected = ids[lengths[ids] > t]
            positions = offsets[selected] + t
            if len(positions):
                curve.append({"t": t, "q_mean": stats(q[:, positions].mean(0)),
                              "actual_rtg": stats(b["return_to_go"][positions])})
        out["first100_q_curves"][side] = curve
    for step in STEPS:
        q = b[f"q_{step}"]
        assert q.shape == (2, len(b["observations"])) and np.isfinite(q).all()
        by_route = {}
        for side in order:
            ids = np.flatnonzero(routes == side)
            rows = {}
            for t in (0, 10, 25, 50, 100, 150, 200):
                selected = ids[lengths[ids] > t]
                ix = offsets[selected] + t
                if len(ix):
                    rows[str(t)] = {"q_mean": stats(q[:, ix].mean(0)),
                                    "q1": stats(q[0, ix]), "q2": stats(q[1, ix]),
                                    "reference_250k_rtg": stats(b["return_to_go"][ix])}
            by_route[side] = rows
        out["cross_checkpoint"][str(step)] = by_route
    teacher_path = folder / "teacher_probe.npz"
    if teacher_path.exists():
        with np.load(teacher_path, allow_pickle=False) as raw:
            a = {k: raw[k] for k in raw.files}
        outcomes = a["outcomes"].astype(str)
        repeats, candidates = outcomes.shape
        q = a["q"].mean(0)
        assert a["q"].shape == (2, candidates)
        assert a["returns"].shape == a["bootstrap_returns"].shape == outcomes.shape
        weights = {"proposal": np.full(candidates, 1 / candidates),
                   "q_only": a["q_only_weights"], "full_teacher": a["weights"]}
        teacher = {"candidate_count": candidates, "continuations": repeats,
                   "route_order": ordered_routes(task, set(outcomes.flat)),
                   "routes": {}, "ess": {},
                   "note": "One 64-candidate cloud at identical initial state, paired continuation RNG. Route is probabilistic after first action.",
                   "q_range": [float(q.min()), float(q.max())]}
        for label, w in weights.items():
            assert np.isfinite(w).all() and (w >= 0).all()
            np.testing.assert_allclose(w.sum(), 1, atol=1e-5)
            teacher["ess"][label] = float(1 / np.square(w).sum())
        for side in teacher["route_order"]:
            mask = outcomes == side
            prob = mask.mean(0)
            row = {}
            for label, w in weights.items():
                mass = float(w @ prob)
                denom = repeats * mass
                row[label] = {
                    "route_mass": mass, "repeat_route_mass": (mask @ w).tolist(),
                    "route_conditional_q_mean": float((w * prob) @ q / mass) if mass else None,
                    "route_conditional_q1": float((w * prob) @ a["q"][0] / mass) if mass else None,
                    "route_conditional_q2": float((w * prob) @ a["q"][1] / mass) if mass else None,
                    "route_conditional_log_proposal": float((w * prob) @ a["logq"] / mass) if mass else None,
                    "route_conditional_actual_return": float((mask * a["returns"] * w).sum() / denom) if mass else None,
                    "route_conditional_bootstrap_return": float((mask * a["bootstrap_returns"] * w).sum() / denom) if mass else None}
            teacher["routes"][side] = row
        for label in weights:
            np.testing.assert_allclose(sum(row[label]["route_mass"] for row in teacher["routes"].values()), 1, atol=1e-5)
        out["teacher_probe"] = teacher
        out["input_sha256"][teacher_path.name] = sha(teacher_path)
    else:
        out["teacher_probe"] = None
    return out


def plots(data, destination):
    fig, axes = plt.subplots(len(data), 3, figsize=(16, 4.5 * len(data)), squeeze=False)
    for row_axes, task in zip(axes, data):
        sides = task["routes"]
        x = np.arange(len(sides))
        ax = row_axes[0]
        for j, side in enumerate(sides):
            r = task["same_initial"][side]
            ax.bar(j, r["q_mean"]["mean"], yerr=r["q_mean"]["sd"] or 0,
                   width=.6, color=COLORS[side], alpha=.8, capsize=4)
            ax.scatter([j - .15, j + .15], [r["q1"]["mean"], r["q2"]["mean"]],
                       color="black", marker="_", s=80)
        ax.set_xticks(x, [f"{s.title()} (n={task['same_initial'][s]['n']})\nMC G={task['same_initial'][s]['actual_rtg']['mean']:.1f}" for s in sides], fontsize=9)
        ax.set_title(f"{task['task']} | 250k Q at identical initial state")
        ax.set_ylabel("Mean twin Q; error bars = action-sample SD")
        ax.text(.02, .98, "Black marks: Q1/Q2 means\nRoutes assigned after full rollout", transform=ax.transAxes, va="top", fontsize=8)
        ax = row_axes[1]
        for side in sides:
            ys = [task["cross_checkpoint"][str(s)][side]["0"]["q_mean"]["mean"] for s in STEPS]
            ax.plot(np.array(STEPS) / 1000, ys, "-o", color=COLORS[side], label=side.title())
        ax.set(title="Later critics on the same 250k initial actions", xlabel="Critic checkpoint (k transitions)", ylabel="Mean twin Q(s0, a0)")
        ax.legend(fontsize=8)
        ax = row_axes[2]
        for side in sides:
            curve = task["first100_q_curves"][side]
            ts = np.array([r["t"] for r in curve])
            means = np.array([r["q_mean"]["mean"] for r in curve])
            sds = np.array([r["q_mean"]["sd"] or 0 for r in curve])
            ax.plot(ts, means, color=COLORS[side], label=side.title())
            ax.fill_between(ts, means - sds, means + sds, color=COLORS[side], alpha=.13)
        ax.set(title="250k critic along reference rollouts", xlabel="Episode time step (different visited states)", ylabel="Mean twin Q; shading = episode SD", xlim=(0, 100))
        for ax in row_axes:
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen-checkpoint inference only | Direct-policy reference rollouts at 250k", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(destination / "q_modes.png", dpi=180)
    plt.close(fig)
    ready = [d for d in data if d["teacher_probe"]]
    if not ready:
        return
    fig, axes = plt.subplots(len(ready), 2, figsize=(12, 4.1 * len(ready)), squeeze=False)
    labels = [("proposal", "Proposal", "#9b9fa6"), ("q_only", "Q-only weights", "#53a587"),
              ("full_teacher", "Full teacher (Q - log q)", "#7958a7")]
    for axs, task in zip(axes, ready):
        teacher = task["teacher_probe"]
        sides = teacher["route_order"]
        x = np.arange(len(sides))
        for j, (key, label, color) in enumerate(labels):
            positions = x + (j - 1) * .25
            masses = [teacher["routes"][side][key]["route_mass"] for side in sides]
            axs[0].bar(positions, masses, width=.24, label=label, color=color)
            vals = [teacher["routes"][side][key]["route_conditional_q_mean"] for side in sides]
            for pos, val in zip(positions, vals):
                if val is not None:
                    axs[1].bar(pos, val, width=.24, color=color)
        for ax in axs:
            ax.set_xticks(x, [s.title() for s in sides])
            ax.grid(axis="y", alpha=.2)
        axs[0].set(title=f"{task['task']} | Induced route mass after first action", ylabel="Weighted route probability", ylim=(0, 1))
        axs[1].set(title="Initial-action Q conditional on resulting route", ylabel="Weighted mean twin Q")
        axs[0].legend(fontsize=8)
    fig.suptitle("One candidate cloud per task | Same-state first action + paired stochastic continuation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(destination / "teacher_route_mass.png", dpi=180)
    plt.close(fig)


def markdown(data, missing):
    lines = ["# OptiQ AntMaze 경로별 Q·teacher 분석", "",
             "학습·모델 수정 없이 저장 체크포인트를 읽은 추론 결과다. 기본 분석은 250k 정책의 상태·행동·실제 보상 기록이며, 500k/750k critic에는 **동일한 250k 상태·행동**을 넣었다.", ""]
    if missing:
        lines += [f"미수집 task: {', '.join(missing)}. 아래는 수집된 task만의 부분 보고다.", ""]
    for task in data:
        lines += [f"## {task['task']}", "",
                  f"학습 source `{task['training_source']}`. 추론 완료 검증: `{task['probe_verification'] is not None}`.", "",
                  "### 같은 초기 상태에서 경로별 Q", "",
                  "모든 rollout은 같은 초기 full state에서 시작한다. 표의 경로는 **실제로 나중에 방문한 경로로 사후 분류**했으므로, 첫 행동 자체가 해당 경로를 결정한다는 뜻은 아니다. Q는 twin critic 평균이다.", "",
                  "| 나중에 방문한 경로 | 표본 | Q(s0,a0), 평균±SD | Q1 / Q2 평균 | 실제 G0, 평균±SD | 성공 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for side in task["routes"]:
            r = task["same_initial"][side]
            lines.append(f"| {side} | {r['n']} | {mean_sd(r['q_mean'])} | {fmt(r['q1']['mean'])} / {fmt(r['q2']['mean'])} | {mean_sd(r['actual_rtg'])} | {r['successes']}/{r['n']} |")
        overall = task["same_initial_all_episodes"]
        lines += ["", f"전체 표본 평균: Q0={fmt(overall['q_mean']['mean'])}, 실현 G0={fmt(overall['actual_rtg']['mean'])}, G0−Q0={fmt(overall['actual_rtg_minus_q']['mean'])}. 이는 같은 시점 정책의 실현 보상과 비교한 수치다. 그 자체로 좌우 Q 편향이나 경로 집중 원인을 입증하지는 않는다."]
        lines += ["", "### 동일 250k 상태·행동 bank를 후기 critic으로 재평가", "",
                  "**후기 Q − 과거 정책의 실현 return을 critic 오차라고 부르지 않는다.** 후기 critic은 후기 정책의 continuation 가치에 대응하며, 이 표는 고정된 입력에 대한 가치 추정의 변화다. t>0에서는 경로별 방문 상태 자체가 다르다.", "",
                  "| 참조 episode t | 경로 | Q:250k critic | Q:500k critic | Q:750k critic | 참조250k RTG |",
                  "|---:|---|---:|---:|---:|---:|"]
        for t in (0, 25, 50, 100, 200):
            for side in task["routes"]:
                rows = [task["cross_checkpoint"][str(s)][side].get(str(t)) for s in STEPS]
                if all(rows):
                    vals = " | ".join(fmt(r["q_mean"]["mean"]) for r in rows)
                    lines.append(f"| {t} | {side} | {vals} | {fmt(rows[0]['reference_250k_rtg']['mean'])} |")
        teacher = task["teacher_probe"]
        lines += ["", "### 초기 상태 teacher 후보와 경로 질량", ""]
        if teacher is None:
            lines += ["teacher_probe.npz가 아직 없어 teacher 비교는 미완료다.", ""]
            continue
        lines += [f"후보{teacher['candidate_count']}개, 동일 후보당 continuation seed{teacher['continuations']}개. 첫 행동만 후보마다 달리하고 이후 continuation 난수는 후보 간 공유했다. 독립적인 teacher cloud를 여러 번 추출한 실험은 아니다.", "",
                  "| 경로 | proposal 질량 | Q-only 질량 | full teacher 질량 | proposal 조건부 Q | full teacher 조건부 Q |",
                  "|---|---:|---:|---:|---:|---:|"]
        for side in teacher["route_order"]:
            r = teacher["routes"][side]
            mass = " | ".join(f"{100*r[k]['route_mass']:.2f}%" for k in ("proposal", "q_only", "full_teacher"))
            lines.append(f"| {side} | {mass} | {fmt(r['proposal']['route_conditional_q_mean'])} | {fmt(r['full_teacher']['route_conditional_q_mean'])} |")
        lines += ["", "조건부 Q는 `Σ w_i P(route|action_i) Q_i / Σ w_i P(route|action_i)`로 계산했다. Full teacher는 Q와 proposal density 보정을 모두 포함한다. 따라서 Q-only와 full teacher의 경로 질량이 다를 수 있다.", "",
                  f"후보 ESS: proposal={fmt(teacher['ess']['proposal'])}, Q-only={fmt(teacher['ess']['q_only'])}, full teacher={fmt(teacher['ess']['full_teacher'])}.", ""]
    lines += ["## 해석 범위", "",
              "- 첫100step Q곡선은 동일한 분기 상태의 반사실적 비교가 아니다. 각 정책 rollout이 실제로 방문한 서로 다른 상태의 값이다.",
              "- 실제 RTG는 gamma=.99로 계산한 유한 episode 보상합이다. critic은 time limit에서 bootstrap하므로 horizon 가까운 Q와 RTG의 차이를 그대로 추정오차라 부를 수 없다.",
              "- teacher probe의 bootstrap_return에는 같은 critic이 포함된다. 독립적인 Monte Carlo 정답이 아니다. 원래 return과 별도 JSON 필드로 보존했다.",
              "- 한 번의 torque로 장기 경로가 고정되지는 않는다. teacher route mass는 해당 첫 행동 이후 continuation까지 포함한 유도 확률이다.",
              "- 사후 경로 분류 표본과 한 candidate cloud만으로 경로 붕괴의 인과 원인을 단정하지 않는다.",
              "- 표와 그림은 한 training seed의 분석이며, 시드 간 통계가 아니다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=base / "remote" if (base / "remote").exists() else base)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", default=["v3", "v4"], choices=["v3", "v4"])
    args = parser.parse_args()
    output = args.output or (args.root.parent / "report" if args.root.name == "remote" else args.root / "report")
    output.mkdir(parents=True, exist_ok=True)
    data, missing = [], []
    for task in args.tasks:
        folder = args.root / task
        if not (folder / "summary.json").exists() or not (folder / "reference_state_action_bank.npz").exists():
            missing.append(task)
        else:
            data.append(process_task(folder))
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_script_sha256": sha(__file__), "missing_tasks": missing,
              "complete_teacher_tasks": [x["task"] for x in data if x["teacher_probe"] and x["probe_verification"]],
              "tasks": data}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "REPORT_KO.md").write_text(markdown(data, missing))
    if data:
        plots(data, output)
    print(json.dumps({"output": str(output), "available_tasks": [d["task"] for d in data],
                      "missing_tasks": missing, "complete_teacher_tasks": result["complete_teacher_tasks"]}))


if __name__ == "__main__":
    main()
