"""Bounded, predeclared poor-performance stopping for the four finite runs.

Only stops explicitly manifested finite-screen services. Saves evidence before
SIGINT. Does not restart runs, delete results, or control other experiments.
These thresholds are compute-budget heuristics, not statistical tests.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/v2_improvement"


def decision(steps, current, reference_v2, reference_optiq):
    if len(steps) < 5 or steps[-1] < 100000:
        return {"stop": False, "reason": "before_100k_or_insufficient_evaluations"}
    recent = np.asarray(current[-5:], dtype=float)
    reference = min(float(np.mean(reference_v2[-5:])), float(np.mean(reference_optiq[-5:])))
    if reference <= 0 or not np.isfinite(recent).all() or not np.isfinite(reference):
        return {"stop": False, "reason": "requires_manual_data_review"}
    mean = float(recent.mean())
    slope = float(np.polyfit(np.asarray(steps[-5:], dtype=float), recent, 1)[0])
    ratio = mean/reference
    stop = (ratio < .5 and slope <= 0) or (steps[-1] >= 200000 and ratio < .6)
    return {"stop": bool(stop), "reason": "substantially_below_both_references" if stop else "continue",
            "step": int(steps[-1]), "current_last5_mean": mean, "conservative_reference_last5_mean": reference,
            "ratio": ratio, "last5_slope_per_step": slope}


def evaluations(directory):
    with np.load(next(Path(directory).glob("eval/*/evaluations.npz"))) as data:
        steps, returns = data["timesteps"].copy(), data["results"].copy()
    if returns.shape != (len(steps), 10) or not np.isfinite(returns).all():
        raise ValueError("Expected ten finite evaluation returns per timestep")
    return {int(step): float(row.mean()) for step, row in zip(steps, returns)}


def main():
    manifest = json.loads((BASE/"finite_screen_manifest.json").read_text())
    assert [r["seed"] for r in manifest["runs"]] == [0, 1, 2, 3]
    refs = json.loads((BASE/"confirmation_manifest.json").read_text())["runs"]
    references = {(r["method"], r["seed"]): evaluations(r["directory"]) for r in refs}
    output = BASE/"finite_screen_monitor.json"
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "decisions": [], "stops": [], "complete": False}
    handled = set()
    while True:
        alive = False
        for run in manifest["runs"]:
            service = f'optiq-v2-finite:optiq-v2-finite-{run["seed"]}'
            assert run["service"] == service
            response = subprocess.run(["supervisorctl", "status", service], capture_output=True, text=True)
            tokens = response.stdout.split()
            status = tokens[1] if len(tokens) >= 2 else "UNKNOWN"
            if status in {"STARTING", "RUNNING", "STOPPING", "BACKOFF", "UNKNOWN"}:
                alive = True
            if status != "RUNNING" or run["seed"] in handled:
                continue
            try:
                current = evaluations(run["directory"])
                v2, optiq = references[("v2", run["seed"])], references[("optiq", run["seed"])]
                steps = sorted(set(current)&set(v2)&set(optiq))
                d = decision(steps, [current[t] for t in steps], [v2[t] for t in steps], [optiq[t] for t in steps])
            except (OSError, ValueError, EOFError, StopIteration) as error:
                print(f"seed {run['seed']}: defer transient/read review: {error}", flush=True)
                continue
            d.update(seed=run["seed"], checked_utc=datetime.now(timezone.utc).isoformat())
            previous = [x for x in state["decisions"] if x["seed"] == run["seed"]]
            if not previous or previous[-1].get("step") != d.get("step"):
                state["decisions"].append(d)
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2)); temporary.replace(output)
            if d["stop"]:
                # Require the periodic checkpoint matching this comparison horizon.
                checkpoint_step = (d["step"]//50000)*50000
                if not list(Path(run["directory"]).glob(f"checkpoints/*/actor_state_{checkpoint_step}.msgpack")):
                    continue
                print(json.dumps({"stopping": service, **d}), flush=True)
                result = subprocess.run(["supervisorctl", "stop", service], capture_output=True, text=True)
                state["stops"].append({**d, "service": service, "returncode": result.returncode,
                                       "supervisor_response": result.stdout.strip()})
                if result.returncode == 0:
                    handled.add(run["seed"])
                temporary.write_text(json.dumps(state, indent=2)); temporary.replace(output)
        if not alive:
            state.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
            output.write_text(json.dumps(state, indent=2))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
