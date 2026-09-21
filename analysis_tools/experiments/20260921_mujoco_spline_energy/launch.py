"""Submit the committed smoke or the 15-run array without duplicates."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

CAMPAIGN = Path("/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921")


def capture(command):
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "main"])
    args = parser.parse_args()
    manifest = json.loads((CAMPAIGN / "manifest/campaign.json").read_text())
    source = CAMPAIGN / "snapshot"
    if capture(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() != manifest["git_commit"]:
        raise RuntimeError("Frozen source commit changed")
    if capture(["git", "-C", str(source), "status", "--porcelain"]).strip():
        raise RuntimeError("Frozen source is dirty")
    receipt = CAMPAIGN / "manifest" / f"launch_{args.stage}.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    job_name = "spline-mj-smoke" if args.stage == "smoke" else "spline-mj-1m"
    if capture(["squeue", "-h", "-u", os.environ["USER"], "-n", job_name]).strip():
        raise RuntimeError(f"Refusing duplicate live job {job_name}")
    if args.stage == "main":
        smoke = json.loads((CAMPAIGN / "checks/gpu_smoke.json").read_text())
        if not smoke["passed"]:
            raise RuntimeError("GPU smoke did not pass")
        if any((CAMPAIGN / "outputs" / f"{env[:-3].lower()}-spline-energy-s{seed}").exists()
               for env in manifest["environments"] for seed in manifest["seeds"]):
            raise RuntimeError("A planned main output already exists")
    diagnostics = {
        "whoami": capture(["whoami"]),
        "sinfo": capture(["sinfo", "-o", "%20P %8a %10l %6D %12T %G"]),
        "squeue": capture(["squeue", "-u", os.environ["USER"], "-o",
                            "%.18i %.18P %.12q %.30j %.2t %.10M %.4D %R"]),
    }
    script_name = "smoke.sbatch" if args.stage == "smoke" else "train.sbatch"
    script = source / "analysis_tools/experiments/20260921_mujoco_spline_energy" / script_name
    job_id = capture(["sbatch", "--parsable", str(script)]).strip().split(";")[0]
    record = {"submitted_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
              "job_id": job_id, "source_commit": manifest["git_commit"],
              "command": ["sbatch", "--parsable", str(script)], "diagnostics": diagnostics}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": args.stage, "job_id": job_id}, indent=2))


if __name__ == "__main__":
    main()
