import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("finite_monitor", Path(__file__).resolve().parents[1]/"scripts/monitor_v2_finite.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_monitor_does_not_stop_warmup_or_recovering_100k_run():
    assert not module.decision([10000]*5, [10]*5, [100]*5, [100]*5)["stop"]
    steps = np.arange(80000, 100001, 5000)
    assert not module.decision(steps, [10, 20, 25, 30, 40], [100]*5, [100]*5)["stop"]


def test_monitor_requires_large_deficit_to_both_and_applies_200k_budget():
    steps = np.arange(80000, 100001, 5000)
    assert module.decision(steps, [40, 35, 30, 25, 20], [100]*5, [100]*5)["stop"]
    assert not module.decision(steps, [40, 35, 30, 25, 20], [100]*5, [40]*5)["stop"]
    assert module.decision(steps+100000, [40, 45, 50, 55, 59], [100]*5, [100]*5)["stop"]
