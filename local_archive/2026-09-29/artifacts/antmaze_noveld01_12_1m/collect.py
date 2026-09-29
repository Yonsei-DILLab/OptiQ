"""Collect the twelve new runs; preserve old and v1-control provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess
import sys

SOURCE = "0d23377f31bdd5c360e9c79d44bb9f4b1341012b"
REMOTE = "/home/heechan/optiq-experiments/antmaze-v234-noveld01-1m-s0-20260922"
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-completed", action="store_true")
    a = p.parse_args()
    from antmaze.multimodal import collect_dense_noveld as collect
    collect.BASE = OUT
    collect.CODE = collect.CODE.replace(collect.ROOT, REMOTE)
    collect.ROOT = REMOTE
    with ThreadPoolExecutor(2) as pool:
        snapshots = dict(pool.map(collect.collect, ("vast-heechan-180", "vast-heechan-199")))
    (OUT / "latest.json").write_text(json.dumps(dict(
        collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), hosts=snapshots), indent=2) + "\n")
    for host, snapshot in snapshots.items():
        assert snapshot["manifest.json"]["source_commit"] == SOURCE
        assert snapshot["manifest.json"]["coefficient"] == .1
        status = snapshot["status.json"]
        print(host, {k:status[k] for k in ("completed", "running", "pending", "failed")})
        if not a.archive_completed:
            continue
        from antmaze.multimodal.dense_noveld_report import verify_run
        for job in status["jobs"]:
            if job["status"] != "completed":
                continue
            name = job["id"]
            assert name == f"{job['task']}-{job['method']}-s0"
            dest = OUT / "runs" / name
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-az", "--checksum", "--exclude=wandb/", "--exclude=train_log/",
                f"{host}:{REMOTE}/runs/{name}/", str(dest) + "/"], check=True)
            proof = verify_run(dest, expected_source=SOURCE, expected_profile="v234-noveld01-1m", expected_coefficient=.1)
            (dest / "archive-verification.json").write_text(json.dumps(dict(
                **proof, host=host, remote=f"{REMOTE}/runs/{name}"), indent=2) + "\n")
            print("ARCHIVED AND VERIFIED", name, flush=True)


if __name__ == "__main__":
    main()
