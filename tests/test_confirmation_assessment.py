import numpy as np
import pytest

from scripts.assess_v2_confirmation import FINAL_STEPS, final_window, drawdown


def complete_runs():
    return {(seed, method): {"steps": FINAL_STEPS.copy(),
            "returns": np.full(len(FINAL_STEPS), (2 if method == "v2" else 1) * (seed+1), dtype=float)}
            for seed in range(4) for method in ("v2", "optiq")}


def test_final_comparison_requires_every_seed_and_every_fixed_checkpoint():
    runs = complete_runs()
    full = final_window(runs)
    assert full["available"] and full["methods"]["v2"]["mean"] == 5.
    assert full["methods"]["optiq"]["mean"] == 2.5
    # One stopped seed cannot be replaced by its favorable available tail.
    runs[3, "v2"] = {"steps": FINAL_STEPS[:-1], "returns": np.full(len(FINAL_STEPS)-1, 1000.)}
    partial = final_window(runs)
    assert not partial["available"] and "methods" not in partial
    assert partial["missing_checkpoints"] == {"v2_seed3": [1000000]}
    del runs[3, "v2"]
    with pytest.raises(ValueError):
        final_window(runs)


def test_drawdown_reports_recovery_and_censors_unfinished_recovery():
    steps = np.arange(80000, 180000, 10000)
    returns = np.array([100., 100., 100., 50., 50., 50., 50., 100., 100., 100.])
    result = drawdown(steps, returns)
    assert result["maximum_drawdown_fraction"] == .5
    assert result["peak_step"] == 100000 and result["trough_step"] == 130000
    assert result["recovery_step"] == 170000 and result["peak_to_recovery_steps"] == 70000
    partial = drawdown(steps[:-1], returns[:-1])
    assert partial["recovery_step"] is None and partial["peak_to_recovery_steps"] is None
    assert partial["unrecovered_elapsed_steps"] == 60000
