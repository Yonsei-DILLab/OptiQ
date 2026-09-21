"""Committed GPU smoke: real Hopper and Humanoid replay/TD interaction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    destinations = []
    for env, steps in [("Hopper-v4", 5400), ("Humanoid-v4", 5250)]:
        output = args.campaign / "smoke" / env
        command = [sys.executable, str(here / "train.py"), "--env", env, "--seed", "1729",
                   "--output", str(output), "--total-steps", str(steps), "--warmup", "5000",
                   "--eval-interval", str(steps), "--eval-episodes", "1",
                   "--checkpoint-interval", str(steps), "--buffer-size", "10000",
                   "--wandb-mode", "disabled", "--require-gpu"]
        subprocess.run(command, check=True, env=os.environ.copy())
        completed = json.loads((output / "COMPLETE.json").read_text())
        assert completed["env_steps"] == steps and completed["optimizer_steps"] == steps - 5000
        destinations.append({"env": env, "output": str(output), "result": completed})
    result = {"passed": True, "runs": destinations,
              "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    (args.campaign / "checks" / "gpu_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

