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


def curves(value, end=250000):
    return {s: {t: value for t in range(230000, end+1, 5000)} for s in range(4)}


def test_strong_group_review_needs_all_seeds_at_fixed_horizon():
    assert not module.strong_group_decision(curves(500, 245000), curves(1000), curves(1200), 250000, .7)["available"]
    missing = curves(500); missing.pop(3)
    assert not module.strong_group_decision(missing, curves(1000), curves(1200), 250000, .7)["available"]


def test_strong_group_review_requires_deficit_to_both_references():
    decision = module.strong_group_decision(curves(600), curves(1400), curves(1000), 250000, .7)
    assert decision["stop"] and len(decision["seed_means"]["finite"]) == 4
    assert decision["tail_steps"] == [230000, 235000, 240000, 245000, 250000]
    assert not module.strong_group_decision(curves(800), curves(1400), curves(1000), 250000, .7)["stop"]
    assert not module.strong_group_decision(curves(700), curves(1400), curves(1000), 250000, .7)["stop"]
