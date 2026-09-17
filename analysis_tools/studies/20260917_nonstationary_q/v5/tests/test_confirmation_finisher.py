import numpy as np
import pytest

from scripts.assess_v2_confirmation import EXPECTED_KEYS, FINAL_STEPS
from scripts.finish_v2_confirmation import completion_ready


def completed_inputs(tmp_path):
    items, runs, states = {}, {}, {}
    for seed, method in EXPECTED_KEYS:
        directory = tmp_path / f"{method}-{seed}"
        saved = directory / "checkpoints" / "run"
        saved.mkdir(parents=True)
        for kind in ("actor", "critic"):
            (saved / f"{kind}_state_1000000.msgpack").write_bytes(b"present")
        name = f"{method}-{seed}"
        items[seed, method] = {"directory": str(directory), "supervisor": name}
        runs[seed, method] = {"steps": FINAL_STEPS.copy(), "completed_marker": True}
        states[name] = "EXITED"
    return items, runs, states


def test_finisher_waits_for_live_process_even_with_final_files(tmp_path):
    items, runs, states = completed_inputs(tmp_path)
    assert completion_ready(items, runs, states)
    states["v2-3"] = "RUNNING"
    assert not completion_ready(items, runs, states)
    states["v2-3"] = "STOPPED"
    with pytest.raises(RuntimeError, match="Unexpected training state"):
        completion_ready(items, runs, states)


def test_finisher_rejects_incomplete_exit_without_dropping_seed(tmp_path):
    items, runs, states = completed_inputs(tmp_path)
    runs[2, "v2"]["steps"] = np.delete(FINAL_STEPS, 10)
    with pytest.raises(RuntimeError, match="Incomplete training exit"):
        completion_ready(items, runs, states)
    runs[2, "v2"]["steps"] = FINAL_STEPS.copy()
    from pathlib import Path
    next(Path(items[2, "v2"]["directory"]).glob("checkpoints/*/critic_state_1000000.msgpack")).unlink()
    with pytest.raises(RuntimeError, match="Missing/nonunique final critic"):
        completion_ready(items, runs, states)
    del runs[2, "v2"]
    with pytest.raises(ValueError, match="All eight"):
        completion_ready(items, runs, states)
