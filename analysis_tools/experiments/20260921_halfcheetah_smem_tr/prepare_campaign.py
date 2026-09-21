"""Freeze a clean committed source manifest before SLURM submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if git("branch", "--show-current") != "heejoon":
        raise RuntimeError("experiments must launch from branch heejoon")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("commit the exact experiment source before freezing")
    manifest = args.campaign/"manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    for directory in ("checks", "smoke", "logs", "state", "outputs", "analysis"):
        (args.campaign/directory).mkdir(parents=True, exist_ok=True)
    paths = git("ls-files", "-z").split("\0")
    hashes = {
        path: hashlib.sha256((REPO/path).read_bytes()).hexdigest()
        for path in paths if path
    }
    tasks = [
        {
            "array_index": seed*2+method_index,
            "seed": seed,
            "method": method,
            "status": "planned",
        }
        for seed in range(3)
        for method_index, method in enumerate(("direct", "smem_tr"))
    ]
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "upstream_commit": (REPO/"UPSTREAM_COMMIT").read_text().strip(),
        "environment": "HalfCheetah-v4",
        "dataset": "online MuJoCo transitions; Gymnasium HalfCheetah-v4",
        "methods": ["direct", "smem_tr"],
        "seeds": [0, 1, 2],
        "total_steps": 1_000_000,
        "evaluation": "stochastic-z and zero-z, 10 episodes every 5K steps",
        "primary_metrics": [
            "final stochastic-z return", "final zero-z return",
            "stochastic-z return AUC", "zero-z return AUC",
        ],
        "hypothesis": "GMM40 SMEM+TR narrow mode-seeking components may improve HalfCheetah return",
        "changed_variable": "actor fitting update only",
        "tasks": tasks,
        "source_sha256": hashes,
    }
    manifest.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({key: record[key] for key in (
        "git_commit", "git_branch", "upstream_commit", "environment", "tasks"
    )}, indent=2))


if __name__ == "__main__":
    main()
