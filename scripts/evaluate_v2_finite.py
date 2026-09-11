"""Independent final evaluation under the finite screen's frozen protocol.

Reuses the common evaluation loop and verified reference episodes. Refuses to
evaluate early/stopped candidates as a completed four-seed result. This script
does not control training or declare that the overall research goal is met.
"""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_v2_confirmation import evaluate_episodes, pool_environment_metrics
from scripts.assess_v2_confirmation import seed_statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_result(result, protocol):
    episodes = result["episodes"]
    assert len(episodes) == protocol["episodes_per_model"]
    assert [e["env_seed"] for e in episodes] == protocol["environment_seeds"]
    assert [e["policy_seed"] for e in episodes] == protocol["policy_seeds"]
    returns = np.array([e["return"] for e in episodes])
    assert np.isfinite(returns).all()
    np.testing.assert_allclose(result["mean_return"], returns.mean(), atol=1.e-9, rtol=0)
    assert result["checkpoint_sha256"] == digest(result["checkpoint"])
    assert result["time_limit_episodes"] == sum(e["truncated"] and not e["terminated"] for e in episodes)


def load_references(protocol):
    results = []
    for source in protocol["references"]:
        assert digest(source["path"]) == source["sha256"]
        data = json.loads(Path(source["path"]).read_text())
        historical = "method" in data and data.get("complete") is True
        if not historical:
            assert data["summary"]["complete"]
        for entry in data["results"]:
            result = dict(entry)
            if historical:
                result.update(method="historical", training_seed=result["seed"])
            validate_result(result, protocol)
            results.append(result)
    assert {(r["method"], r["training_seed"]) for r in results} == {
        (method, seed) for method in ("v2", "optiq", "historical") for seed in range(4)}
    assert len(results) == 12
    return results


def completion_gate(manifest, protocol):
    reasons = []
    assert len(manifest["runs"]) == 4 and {r["seed"] for r in manifest["runs"]} == set(range(4))
    for item in manifest["runs"]:
        directory = Path(item["directory"])
        response = subprocess.run(["supervisorctl", "status", item["service"]], capture_output=True, text=True)
        tokens = response.stdout.split()
        status = tokens[1] if len(tokens) >= 2 else "UNKNOWN"
        if status != "EXITED" or not (directory/"completed.json").exists():
            reasons.append({"seed": item["seed"], "reason": "training_not_completed", "status": status})
            continue
        completed = json.loads((directory/"completed.json").read_text())
        if completed.get("timesteps") != protocol["checkpoint_step"]:
            reasons.append({"seed": item["seed"], "reason": "incomplete_step_count", "status": status})
            continue
        cfg = json.loads((directory/"config.json").read_text())
        assert cfg["seed"] == item["seed"] and cfg["runtime"]["git_commit"] == item["commit"]
        assert cfg["alg"]["actor"]["latent_prior"] == "finite"
        assert cfg["total_steps"] == protocol["checkpoint_step"]
        with np.load(next(directory.glob("eval/*/evaluations.npz"))) as data:
            assert np.isin(protocol["primary_window_steps"], data["timesteps"]).all()
            assert data["results"].shape == (len(data["timesteps"]), 10)
            assert np.isfinite(data["results"]).all()
        for kind in ("actor", "critic"):
            assert len(list(directory.glob(f'checkpoints/*/{kind}_state_{protocol["checkpoint_step"]}.msgpack'))) == 1
    return reasons


def summarize(results, references):
    if {r["training_seed"] for r in results} != set(range(4)) or len(results) != 4:
        return {"complete": False}
    by_key = {(r["method"], r["training_seed"]): r for r in results+references}
    methods = {method: seed_statistics([by_key[method, s]["mean_return"] for s in range(4)])
               for method in ("finite", "v2", "optiq", "historical")}
    gaps = {method: seed_statistics([by_key["finite", s]["mean_return"]-by_key[method, s]["mean_return"]
                                     for s in range(4)]) for method in ("v2", "optiq", "historical")}
    return {"complete": True, "methods": methods, "paired_seed_gaps": gaps,
            "scope": "Secondary final-checkpoint evaluation; primary 900k–1M window and stability require separate review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify references and report readiness without evaluating")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT/"outputs/v2_improvement/finite_final_evaluation_protocol.json")
    args = parser.parse_args()
    base = ROOT/"outputs/v2_improvement"
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    assert digest(protocol["common_evaluator"]) == protocol["common_evaluator_sha256"]
    launch = json.loads(Path(protocol.get("launch_protocol", base/"finite_screen_protocol.json")).read_text())
    for name, wanted in launch["core_sha256"].items():
        assert digest(ROOT/name) == wanted
    manifest = json.loads(Path(protocol["candidate_manifest"]).read_text())
    references = load_references(protocol)
    pending = completion_gate(manifest, protocol)
    if args.check:
        print(json.dumps({"ready": not pending, "pending": pending, "verified_reference_models": len(references)}))
        return
    if pending:
        raise RuntimeError(f"Final evaluation requires all four completed runs: {pending}")
    evaluation_lock = Path(protocol.get("evaluation_lock", base/"finite_final_evaluation.lock")).open("a+")
    fcntl.flock(evaluation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    from flax import serialization
    import gymnasium as gym
    import jax
    from omegaconf import OmegaConf
    from optiq_dime import OptiQDIME
    from optiq_dime.latent import FiniteMixtureTrainState

    output = Path(protocol.get("result_path", base/"finite_independent_1000000.json"))
    record = json.loads(output.read_text()) if output.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(protocol_path),
        "evaluator_sha256": digest(__file__), "jax_version": jax.__version__, "gymnasium_version": gym.__version__,
        "devices": [str(d) for d in jax.devices()], "results": [], "summary": {"complete": False}}
    record["candidate_label"] = protocol.get("candidate_label", "Finite-policy candidate")
    assert record["protocol_sha256"] == digest(protocol_path) and record["evaluator_sha256"] == digest(__file__)
    for existing in record["results"]:
        validate_result(existing, protocol)
    for item in sorted(manifest["runs"], key=lambda r: r["seed"]):
        if any(r["training_seed"] == item["seed"] for r in record["results"]):
            continue
        directory = Path(item["directory"])
        cfg = OmegaConf.load(directory/"config.json")
        checkpoint = next(directory.glob('checkpoints/*/actor_state_1000000.msgpack'))
        environment = gym.make(cfg.env_name)
        model = OptiQDIME("MlpPolicy", environment, None, 1, cfg)
        try:
            model.policy.actor_state = serialization.from_bytes(model.policy.actor_state, checkpoint.read_bytes())
            assert isinstance(model.policy.actor_state, FiniteMixtureTrainState)
            episodes = evaluate_episodes(model, environment, protocol["episodes_per_model"], protocol["environment_seeds"][0])
            returns = np.array([e["return"] for e in episodes])
            result = {"method": "finite", "training_seed": item["seed"], "run": directory.name,
                "source_commit": item["commit"], "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "episodes": episodes, "mean_return": float(returns.mean()), "episode_return_sd": float(returns.std(ddof=1)),
                "mean_length": float(np.mean([e["length"] for e in episodes])),
                "time_limit_episodes": sum(e["truncated"] and not e["terminated"] for e in episodes),
                "environment_metrics": pool_environment_metrics(episodes)}
            validate_result(result, protocol)
            record["results"].append(result)
            record["summary"] = summarize(record["results"], references)
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2)); temporary.replace(output)
            print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        finally:
            model.get_env().close()
    print(json.dumps(record["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
