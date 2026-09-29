"""Read-only verification using W&B GraphQL; no training process changes."""
import datetime
import json
import os
import requests
ids = ["gxv5q9i8", "vso9d8ls", "670yefgm", "a72pwt97"]
fields = " ".join(f'r{i}:run(name:"{rid}"){{name state summaryMetrics}}' for i, rid in enumerate(ids))
query = '{project(name:"v5-heechan-gmm",entityName:"OptiQ"){name ' + fields + '}}'
response = requests.post("https://api.wandb.ai/graphql", auth=("api", os.environ["WANDB_API_KEY"]), json={"query":query}, timeout=30)
response.raise_for_status()
data = response.json()
assert not data.get("errors"), data
project = data["data"]["project"]
assert project["name"] == "v5-heechan-gmm"
rows = []
for seed, rid in enumerate(ids):
    run = project[f"r{seed}"]
    assert run["name"] == rid
    summary = json.loads(run["summaryMetrics"])
    rows.append({"seed":seed, "id":rid, "project":project["name"], "url":f"https://wandb.ai/OptiQ/v5-heechan-gmm/runs/{rid}", "state":run["state"], "summary_step":summary.get("_step"), "env_steps":summary.get("time/total_timesteps"), "timestamp":summary.get("_timestamp")})
print(json.dumps({"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs":rows}, indent=2))
