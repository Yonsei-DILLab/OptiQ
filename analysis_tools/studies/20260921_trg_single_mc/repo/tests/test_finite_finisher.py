import hashlib
import json
from pathlib import Path
import pytest
from scripts import finish_v2_finite as finisher


def setup(tmp_path, monkeypatch):
    base = tmp_path/"outputs/v2_improvement"
    base.mkdir(parents=True)
    evaluator = tmp_path/"common_evaluator.py"
    evaluator.write_text("# fixed protocol\n")
    manifest = base/"manifest.json"
    manifest.write_text(json.dumps({"runs": []}))
    protocol = {"candidate_manifest": str(manifest), "common_evaluator": str(evaluator),
                "common_evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest()}
    (base/"finite_final_evaluation_protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(finisher, "ROOT", tmp_path)
    monkeypatch.setattr(finisher, "load_references", lambda _: [])
    return base


@pytest.mark.parametrize("status", ["STOPPED", "EXITED"])
def test_incomplete_seed_never_runs_final_comparison(tmp_path, monkeypatch, status):
    base = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(finisher, "completion_gate", lambda *_: [{"seed": 2, "status": status}])
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not evaluate a screen with an incomplete seed")
    monkeypatch.setattr(finisher.subprocess, "run", forbidden)
    finisher.main([])
    state = json.loads((base/"finite_finisher_status.json").read_text())
    assert state["stage"] == "screen_incomplete" and not state["goal_complete"]


def test_wait_then_report_and_evaluate_without_controlling_training(tmp_path, monkeypatch):
    base = setup(tmp_path, monkeypatch)
    pending = iter([[{"seed": 0, "status": "RUNNING"}], []])
    monkeypatch.setattr(finisher, "completion_gate", lambda *_: next(pending))
    waits = []
    monkeypatch.setattr(finisher.time, "sleep", waits.append)
    commands = []
    def run(command, **kwargs):
        commands.append(Path(command[1]).name)
        if len(commands) == 2:
            (base/"finite_independent_1000000.json").write_text(json.dumps({
                "summary": {"complete": True}, "results": [{"training_seed": i} for i in range(4)]}))
    monkeypatch.setattr(finisher.subprocess, "run", run)
    finisher.main([])
    state = json.loads((base/"finite_finisher_status.json").read_text())
    assert waits == [30]
    assert commands == ["report_v2_finite.py", "evaluate_v2_finite.py"]
    assert state["stage"] == "complete" and not state["goal_complete"]
