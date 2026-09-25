"""Archive and verify the four frozen N-Way OptiQ runs, then plot goal use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_COMMIT = "973eca4a432cf027a5c725a372aafb86b9ed6c83"
REMOTE_ROOT = "/home/heechan/optiq-experiments/nway-wallfree-optiq-1m-s0-20260926"
TASKS = ((8, "vast-heechan-199", "199-gpu3"),
         (12, "vast-heechan-180", "180-gpu0"),
         (16, "vast-heechan-199", "199-gpu3"),
         (32, "vast-heechan-180", "180-gpu0"))


def sync(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--checksum", source, str(destination)], check=True)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def evaluate(run: Path, n: int) -> tuple[list[dict], dict]:
    config = json.loads((run / "config.json").read_text())
    progress = json.loads((run / "progress.json").read_text())
    expected = dict(task=f"{n}way", method="optiq", seed=0,
                    steps=1_000_000, num_envs=16, batch_size=256,
                    updates_per_collect=16, temperature=1.0,
                    source_commit=SOURCE_COMMIT)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"N-Way config mismatch: {run.name}")
    if ((progress.get("status"), progress.get("steps"), progress.get("updates")) !=
            ("complete", 1_000_000, 998_976)):
        raise ValueError(f"incomplete optimizer count: {run.name}")
    final = progress["latest_evaluation"]
    if final["step"] != 1_000_000 or final["updates"] != 998_976:
        raise ValueError(f"final evaluation mismatch: {run.name}")

    rows = []
    for step in range(100_000, 1_000_001, 100_000):
        policy = run / "evaluations" / f"{step:09d}_policy.npz"
        mu = run / "evaluations" / f"{step:09d}_mu_only.npz"
        probe = run / "evaluations" / f"{step:09d}_probe.npz"
        if not all(path.is_file() for path in (policy, mu, probe)):
            raise FileNotFoundError(f"missing evaluation at {step}: {run.name}")
        outcome = dict(step=step)
        for mode, path in (("policy", policy), ("mu_only", mu)):
            with np.load(path) as data:
                goal_ids, returns, xy = data["goal_ids"], data["returns"], data["xy"]
                count = 1024 if step == 1_000_000 else 256
                if (goal_ids.shape != (count,) or xy.shape[0] != count or
                        xy.shape[-1] != 2 or not np.isfinite(returns).all() or
                        np.any((goal_ids < -1) | (goal_ids >= n))):
                    raise ValueError(f"invalid rollout: {path}")
                goals = np.bincount(goal_ids[goal_ids >= 0], minlength=n).tolist()
                outcome[mode] = dict(episodes=count, goals=goals,
                                     reachable_goals=sum(value > 0 for value in goals),
                                     success=sum(goals) / count,
                                     mean_return=float(np.mean(returns)))
        if step == 1_000_000:
            for mode in ("policy", "mu_only"):
                summary = final[mode]
                actual = outcome[mode]
                if (actual["goals"] != summary["goals"] or
                        actual["episodes"] != summary["episodes"] or
                        not np.isclose(actual["success"], summary["success"])):
                    raise ValueError(f"final summary disagrees with rollout: {run.name}")
        rows.append(outcome)

    replay = run / "checkpoints" / "replay_001000000.npz"
    policy = run / "checkpoints" / "policy_001000000.msgpack"
    figure = run / "figures" / "001000000_policy_q_trajectories.png"
    if not policy.is_file() or not figure.is_file():
        raise FileNotFoundError(f"final checkpoint/figure missing: {run.name}")
    with zipfile.ZipFile(replay) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt replay: {replay}")
    return rows, dict(policy_checkpoint_sha256=digest(policy),
                      replay_sha256=digest(replay), figure_sha256=digest(figure))


def render(results: dict, destination: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    for column, (n, _, _) in enumerate(TASKS):
        task = f"{n}way"
        upper, lower = axes[:, column]
        if task not in results:
            upper.set_title(f"{n}-Way · pending")
            for ax in (upper, lower):
                ax.axis("off")
            continue
        rows = results[task]["evaluations"]
        upper.plot([row["step"] for row in rows],
                   [row["policy"]["reachable_goals"] for row in rows],
                   marker="o", label="Direct policy")
        upper.plot([row["step"] for row in rows],
                   [row["mu_only"]["reachable_goals"] for row in rows],
                   marker="o", label="Random-z μ-only")
        upper.set(xlabel="Environment transitions", ylabel="Goals reached",
                  title=f"{n}-Way · seed 0", ylim=(0, n + .5))
        upper.grid(alpha=.2)
        counts = rows[-1]["policy"]["goals"]
        lower.bar(np.arange(n), counts, width=.8)
        lower.set(xlabel="Goal index", ylabel="Final episodes (n=1024)",
                  title=f"Final: {sum(value > 0 for value in counts)}/{n} goals")
        lower.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Wall-free N-Way OptiQ · sampled-policy goal coverage · 1M steps")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for n, host, shard in TASKS:
        meta = root / "hosts" / host
        sync(f"{host}:{REMOTE_ROOT}/manifest.json", meta / "manifest.json")
        sync(f"{host}:{REMOTE_ROOT}/queue-{shard}.json", meta / f"queue-{shard}.json")
        manifest = json.loads((meta / "manifest.json").read_text())
        state = json.loads((meta / f"queue-{shard}.json").read_text())
        if manifest["source_commit"] != SOURCE_COMMIT or state["source_commit"] != SOURCE_COMMIT:
            raise ValueError(f"source mismatch on {host}")
        name = f"{n}way-optiq-s0"
        if f"{n}way:optiq" not in state.get("completed", []):
            continue
        sync(f"{host}:{REMOTE_ROOT}/jobs/{name}.json", meta / f"{name}.json")
        job = json.loads((meta / f"{name}.json").read_text())
        if job["status"] != "complete" or job["source_commit"] != SOURCE_COMMIT:
            raise ValueError(f"completed queue/job mismatch: {name}")
        run = root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        sync(f"{host}:{REMOTE_ROOT}/runs/{name}/", run)
        rows, hashes = evaluate(run, n)
        results[f"{n}way"] = dict(host=host, evaluations=rows, hashes=hashes,
                                   archived_files={str(path.relative_to(run)): digest(path)
                                                   for path in sorted(run.rglob("*"))
                                                   if path.is_file()})
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                     cwd=Path(__file__).resolve().parents[1]).strip()
    archive = dict(training_source_commit=SOURCE_COMMIT, reporting_source_commit=source,
                   completed=len(results), expected=len(TASKS), runs=results)
    (root / "archive-manifest.json").write_text(json.dumps(archive, indent=2) + "\n")
    render(results, root / "figures" / "goal_coverage.png")
    print(json.dumps(dict(completed=len(results), expected=len(TASKS), output=str(root))))


if __name__ == "__main__":
    main()
