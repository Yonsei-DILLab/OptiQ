"""Freeze one clean committed source and a 15-run campaign manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=SOURCE, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if git("branch", "--show-current") != "heejoon":
        raise RuntimeError("AGENTS.md requires experiment commits on branch heejoon")
    if git("status", "--porcelain"):
        raise RuntimeError("Commit the exact source/config/launcher before freezing")
    if args.campaign.exists():
        raise FileExistsError(args.campaign)
    for name in ["checks", "smoke", "logs", "state", "outputs", "analysis", "manifest"]:
        (args.campaign / name).mkdir(parents=True, exist_ok=True)
    snapshot = args.campaign / "snapshot"
    subprocess.run(["git", "clone", "--no-hardlinks", str(SOURCE), str(snapshot)], check=True)
    commit = git("rev-parse", "HEAD")
    subprocess.run(["git", "checkout", "--detach", commit], cwd=snapshot, check=True)
    files = git("ls-files").splitlines()
    tasks = [{"array_index": i * 3 + seed, "env": env, "seed": seed}
             for i, env in enumerate(ENVS) for seed in range(3)]
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "project_name": "OptiQ MuJoCo Spline Energy", "git_commit": commit,
              "git_branch": "heejoon", "environments": ENVS, "seeds": [0, 1, 2],
              "total_runs": 15, "total_steps_per_run": 1_000_000,
              "hypothesis": "one state-conditioned spline circuit can jointly represent Q,V,policy with exact one-forward sampling",
              "baseline": "existing Direct GMM/TRG MuJoCo protocol; not retrained in this launch",
              "changed_variable": "actor and critic replaced together by unified conditional spline circuit",
              "primary_metric": "eval/mean_reward; 10 native-policy episodes every 10k steps",
              "tasks": tasks,
              "source_sha256": {p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() for p in files}}
    (args.campaign / "manifest" / "campaign.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["git_commit", "total_runs", "tasks"]}, indent=2))


if __name__ == "__main__":
    main()

