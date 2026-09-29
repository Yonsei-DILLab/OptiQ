"""Read this single run; optionally archive and verify its completed result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SOURCE = "a4ea6c1e3284199bbfab7057282fdc57fd3742a2"
NAME = "antmaze-v1-dipo-noveld001-1m-s0-20260922"
HOST = "vast-heechan-180"
REMOTE = "/home/heechan/optiq-experiments/" + NAME
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-completed", action="store_true")
    args = parser.parse_args()
    script = f'''import json
from pathlib import Path
r=Path({REMOTE!r})
files=['manifest.json','registration.json','status.json','failure.json','preflight-proof.json','result.json',
       'runs/v1-dipo-s0/config.json','runs/v1-dipo-s0/progress.json','runs/v1-dipo-s0/wandb.json',
       'runs/v1-dipo-s0/failure.json','runs/v1-dipo-s0/verification.json']
print(json.dumps({{name:json.loads((r/name).read_text()) for name in files if (r/name).exists()}}))
'''
    reply = subprocess.run(["ssh", HOST, "python3 -"], input=script, text=True, capture_output=True, check=True)
    records = json.loads(reply.stdout)
    for name, value in records.items():
        dest = ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(value, indent=2) + "\n")
    if args.archive_completed and records.get("result.json", {}).get("completed"):
        subprocess.run(["rsync", "-az", "--exclude=wandb/", "--exclude=train_log/",
            f"{HOST}:{REMOTE}/runs/", str(ROOT / "runs") + "/"], check=True)
        sys.path.insert(0, str(ROOT.parents[1]))
        from antmaze.multimodal.dense_noveld_report import verify_run
        proof = verify_run(ROOT / "runs/v1-dipo-s0", expected_source=SOURCE,
            expected_profile="v1-dipo-noveld001-1m", expected_coefficient=.01)
        (ROOT / "archive-verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(status=records.get("status.json"),
        progress=records.get("runs/v1-dipo-s0/progress.json"),
        wandb=records.get("runs/v1-dipo-s0/wandb.json")), indent=2))


if __name__ == "__main__":
    main()
