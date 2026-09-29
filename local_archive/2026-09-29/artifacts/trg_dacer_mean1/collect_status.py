"""Read-only snapshot of the six approved mean-head initialization runs."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = "import json,re,subprocess,time\nfrom pathlib import Path\nr=Path(\"/home/heechan/optiq-experiments/trg-dacer-mean1-20260921\")\ndata={\"checked_at\":time.time()}\nfor key in (\"manifest\",\"status\",\"failure\"):\n p=r/(key+\".json\");data[key]=json.loads(p.read_text()) if p.exists() else None\ndata[\"runs\"]=[]\nfor f in sorted((r/\"jobs\").glob(\"*.json\")):\n j=json.loads(f.read_text());out={k:j.get(k) for k in (\"name\",\"task\",\"seed\",\"status\",\"gpu\",\"pid\",\"error\")}\n p=r/\"logs\"/(j[\"name\"]+\".log\")\n if p.exists():\n  log=p.read_text(errors=\"replace\")\n  matches=re.findall(r\"https://wandb.ai/OptiQ/gmm-trg/runs/[A-Za-z0-9]+\",log)\n  out[\"wandb_url\"]=matches[-1] if matches else None\n  steps=re.findall(r\"total_timesteps\\s*\\|\\s*(\\d+)\",log)\n  out[\"step\"]=int(steps[-1]) if steps else None\n  out[\"log_tail\"]=log[-1800:]\n paths=list((r/\"outputs\").glob(j[\"name\"]+\"_*/config.json\"))\n if paths:\n  cfg=json.loads(paths[0].read_text());a=cfg[\"alg\"][\"actor\"]\n  out[\"resolved_config\"]=cfg\n  out[\"config\"]={\"path\":str(paths[0]),\"mean_output_init_scale\":a[\"mean_output_init_scale\"],\"temperature\":a[\"temperature\"],\"beta\":a[\"density_beta\"],\"log_std\":[a[\"log_std_min\"],a[\"log_std_max\"],a[\"initial_log_std\"]],\"dacer\":cfg[\"dacer\"],\"source\":cfg[\"runtime\"],\"total_steps\":cfg[\"total_steps\"]}\n data[\"runs\"].append(out)\ndata[\"gpu\"]=subprocess.check_output([\"nvidia-smi\",\"--query-gpu=index,name,utilization.gpu,memory.used\",\"--format=csv,noheader\"],text=True)\nprint(json.dumps(data))\n"
response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "vast-heechan-199", "python3 -"], input=REMOTE, text=True, capture_output=True,
    timeout=60, check=True)
data = json.loads(response.stdout)
(HERE / "monitor-latest.json").write_text(json.dumps(data, indent=2) + "\n")
for key in ("manifest", "status", "failure"):
    if data.get(key) is not None:
        (HERE / (key + ".json")).write_text(json.dumps(data[key], indent=2) + "\n")
for run in data["runs"]:
    cfg = run.get("resolved_config")
    if cfg:
        folder = HERE / "configs"
        folder.mkdir(exist_ok=True)
        (folder / (run["name"] + ".json")).write_text(json.dumps(cfg, indent=2) + "\n")
print(json.dumps({"status": data["status"], "failure": data["failure"], "gpu": data["gpu"],
    "runs": [{k:r.get(k) for k in ("name", "status", "gpu", "step", "wandb_url", "log_tail")}
             for r in data["runs"]]}, indent=2))

