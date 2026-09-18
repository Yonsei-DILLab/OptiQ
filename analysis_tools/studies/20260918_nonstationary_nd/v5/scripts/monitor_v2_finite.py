"""Poor-performance stopping for the four finite runs.

Only stops explicitly manifested finite-screen services. Saves evidence before
SIGINT. Does not restart runs, delete results, or control other experiments.
These thresholds are compute-budget heuristics, not statistical tests.
The optional stronger-reference group review is explicitly recorded as an
adaptive decision after the 200k results, not as a predeclared launch rule.
"""
from datetime import datetime, timezone
import hashlib
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


def strong_group_decision(candidate, continuous, historical, cutoff, threshold):
    """All four seeds at one fixed horizon, against BOTH stronger references."""
    pending = {"available": False, "stop": False, "reason": "awaiting_all_seeds_at_group_cutoff"}
    if any(set(group) != set(range(4)) for group in (candidate, continuous, historical)):
        return pending
    curves = [group[s] for group in (candidate, continuous, historical) for s in range(4)]
    common = sorted(set.intersection(*(set(curve) for curve in curves)))
    steps = [t for t in common if t <= cutoff]
    if cutoff not in steps or len(steps) < 5:
        return pending
    scores = {name: [float(np.mean([group[s][t] for t in steps[-5:]])) for s in range(4)]
              for name, group in (("finite", candidate), ("v2", continuous), ("historical", historical))}
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    reference = min(means["v2"], means["historical"])
    if not all(np.isfinite(v) for v in means.values()) or reference <= 0:
        return {**pending, "reason": "requires_manual_data_review"}
    return {"available": True, "stop": bool(means["finite"] < threshold*reference),
            "step": cutoff, "tail_steps": steps[-5:], "seed_means": scores, "means": means,
            "ratio_to_lower_strong_reference": means["finite"]/reference, "threshold": threshold,
            "reason": "adaptive_budget_review_against_both_stronger_references"}


def main():
    manifest = json.loads((BASE/"finite_screen_manifest.json").read_text())
    assert [r["seed"] for r in manifest["runs"]] == [0, 1, 2, 3]
    refs = json.loads((BASE/"confirmation_manifest.json").read_text())["runs"]
    references = {(r["method"], r["seed"]): evaluations(r["directory"]) for r in refs}
    policy = json.loads((BASE/"finite_screen_protocol.json").read_text())
    group_review = policy.get("adaptive_strong_reference_review")
    historical = {}
    if group_review:
        items = json.loads((BASE/"historical_behavior010_reference.json").read_text())["runs"]
        for item in items:
            path = next(Path(item["directory"]).glob("eval/*/evaluations.npz"))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["evaluation_sha256"]
            historical[item["seed"]] = evaluations(item["directory"])
    output = BASE/"finite_screen_monitor.json"
    state = json.loads(output.read_text()) if output.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "decisions": [], "stops": [], "complete": False}
    state.setdefault("activations", []).append(datetime.now(timezone.utc).isoformat())
    handled = {r["seed"] for r in state["stops"] if r["returncode"] == 0}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2)); temporary.replace(output)

    while True:
        if group_review and not state.get("strong_reference_review", {}).get("applied", False):
            try:
                group = strong_group_decision(
                    {r["seed"]: evaluations(r["directory"]) for r in manifest["runs"]},
                    {s: references["v2", s] for s in range(4)}, historical,
                    group_review["step"], group_review["threshold"])
                group["checked_utc"] = datetime.now(timezone.utc).isoformat()
                state["strong_reference_review"] = group
                save()
                if group["available"] and not group["stop"]:
                    group["applied"] = True
                    save()
                elif group["available"] and group["stop"]:
                    ready = all(list(Path(r["directory"]).glob(
                        f'checkpoints/*/{kind}_state_{group["step"]}.msgpack'))
                        for r in manifest["runs"] for kind in ("actor", "critic"))
                    if ready:
                        for run in manifest["runs"]:
                            if run["seed"] in handled:
                                continue
                            service = f'optiq-v2-finite:optiq-v2-finite-{run["seed"]}'
                            assert run["service"] == service
                            print(json.dumps({"stopping_group_member": service, **group}), flush=True)
                            result = subprocess.run(["supervisorctl", "stop", service], capture_output=True, text=True)
                            state["stops"].append({"seed": run["seed"], "step": group["step"],
                                "reason": group["reason"], "service": service, "returncode": result.returncode,
                                "supervisor_response": result.stdout.strip()})
                            if result.returncode == 0:
                                handled.add(run["seed"])
                            save()
                        group["applied"] = len(handled) == 4
                        save()
            except (OSError, ValueError, EOFError, StopIteration) as error:
                print(f"group review deferred: {error}", flush=True)
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
